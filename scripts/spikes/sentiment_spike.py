"""Task 1 spike: does Korean-language sentiment classification clear a usable
accuracy bar before Phase 4 wires it into a live trading risk-gate?

Scores a hand-labeled sample of Toss news articles two ways:

1. LLM approach -- a Korean system prompt against OpenAI, reusing the
   structured-output (tool-calling) pattern from
   ``src/gpt_quant_v2/infrastructure/external/openai_client.py``, but as a
   standalone client (no production code imported) so this script has zero
   coupling to the app's config/DI. Needs ``OPENAI_API_KEY`` in the
   environment; if it's missing this approach is skipped and reported as a
   finding, not a blocker.

2. Local-model approach -- a Korean FinBERT sentiment checkpoint
   (``snunlp/KR-FinBert-SC``) loaded via ``transformers``, run fully offline
   after the first download.

Both approaches score the same hand-labeled sample (see
``label_dataset.py`` output / ``001-hand-labels.json``) and accuracy against
the hand labels is computed and printed + written to a JSON results file.

This is a standalone spike script. It does not touch any production code
path and is not imported by the application.

Usage:
    python scripts/spikes/sentiment_spike.py \
        --sample docs/spikes/001-sample.json \
        --labels docs/spikes/001-hand-labels.json \
        --out docs/spikes/001-results.json

Dependencies beyond the repo's own (``httpx`` is already a project dependency):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "transformers>=4.40"
These are spike-only extras, not added to pyproject.toml -- this script is not
imported by the application.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

LABELS = ("BULLISH", "BEARISH", "NEUTRAL")

# ---------------------------------------------------------------------------
# Korean system prompt for this spike (standalone -- NOT written into the
# production prompts.py file). Mirrors the structure/intent of
# SENTIMENT_SYSTEM_PROMPT in openai_client.py's prompts module, translated
# and adapted for Korean financial news.
# ---------------------------------------------------------------------------
KOREAN_SYSTEM_PROMPT = """\
당신은 한국 주식시장(코스피/코스닥) 뉴스를 분석하는 금융 애널리스트입니다.
기사를 읽고 해당 뉴스가 관련 종목 또는 시장 전반에 미치는 영향을 판단하세요.

판단 기준:
1. 일반적인 기사 논조가 아니라 실제 매매에 참고할 수 있는 시그널에 집중할 것
2. 직접적 영향과 간접적 영향을 모두 고려할 것
3. 과거 맥락보다 최근 사건에 더 큰 비중을 둘 것
4. 주가를 움직일 수 있는 구체적 촉매를 식별할 것
5. 객관적으로 판단하고 사전 시장 심리에 휩쓸리지 말 것

다음 JSON 형식으로만 답하세요:
{
    "sentiment_label": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence": <0.0에서 1.0 사이의 확신도>,
    "reasoning": "<판단 근거 한두 문장>"
}

라벨 기준:
- BULLISH: 긍정적 촉매, 성장 동력, 우호적 소식
- BEARISH: 부정적 촉매, 리스크, 불리한 전개
- NEUTRAL: 신호가 혼재되어 있거나, 통상적인 뉴스거나, 방향성이 뚜렷하지 않음
"""

KOREAN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sentiment_analysis",
        "description": "한국 금융 뉴스의 시장 심리(감성)를 분석한다",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment_label": {"type": "string", "enum": list(LABELS)},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["sentiment_label", "confidence", "reasoning"],
        },
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Approach 1: LLM (OpenAI, Korean prompt)
# ---------------------------------------------------------------------------


async def score_with_openai(
    articles: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]] | None:
    """Score articles via OpenAI chat completions + tool calling.

    Returns None (rather than raising) if no API key is configured -- this
    approach is allowed to be a documented finding, not a hard failure.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GPTQ_OPENAI_API_KEY")
    if not api_key:
        return None

    import httpx

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        timeout=httpx.Timeout(60.0),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    ) as client:
        for article in articles:
            user_prompt = f"제목: {article['title']}\n\n본문: {article['text']}"
            payload = {
                "model": model,
                "temperature": 0.1,
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": KOREAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "tools": [KOREAN_TOOL_SCHEMA],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "sentiment_analysis"},
                },
            }
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            tool_call = data["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(tool_call["function"]["arguments"])
            results.append(
                {
                    "id": article["id"],
                    "predicted": args["sentiment_label"].upper(),
                    "confidence": args.get("confidence"),
                    "reasoning": args.get("reasoning"),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Approach 2: Local Korean FinBERT (snunlp/KR-FinBert-SC)
# ---------------------------------------------------------------------------

FINBERT_LABEL_MAP = {
    "negative": "BEARISH",
    "neutral": "NEUTRAL",
    "positive": "BULLISH",
}


def score_with_finbert(
    articles: list[dict[str, Any]],
    checkpoint: str = "snunlp/KR-FinBert-SC",
) -> list[dict[str, Any]] | None:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    results: list[dict[str, Any]] = []
    batch_size = 8
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        texts = [a["text"] for a in batch]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for article, p in zip(batch, probs, strict=True):
            idx = int(p.argmax())
            raw_label = model.config.id2label[idx]
            results.append(
                {
                    "id": article["id"],
                    "predicted": FINBERT_LABEL_MAP.get(raw_label, "NEUTRAL"),
                    "confidence": float(p[idx]),
                    "raw_label": raw_label,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_accuracy(
    predictions: list[dict[str, Any]], hand_labels: dict[str, str]
) -> dict[str, Any]:
    correct = 0
    total = 0
    confusion: dict[str, dict[str, int]] = {label: dict.fromkeys(LABELS, 0) for label in LABELS}
    per_article = []
    for pred in predictions:
        gold = hand_labels.get(pred["id"])
        if gold is None:
            continue
        total += 1
        is_correct = gold == pred["predicted"]
        correct += int(is_correct)
        confusion[gold][pred["predicted"]] += 1
        per_article.append(
            {"id": pred["id"], "gold": gold, "predicted": pred["predicted"], "correct": is_correct}
        )
    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion,
        "per_article": per_article,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("docs/spikes/001-sample.json"))
    parser.add_argument("--labels", type=Path, default=Path("docs/spikes/001-hand-labels.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/spikes/001-results.json"))
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip the OpenAI approach even if a key exists"
    )
    parser.add_argument(
        "--skip-finbert", action="store_true", help="Skip the local FinBERT approach"
    )
    args = parser.parse_args()

    articles = load_json(args.sample)
    hand_labels = load_json(args.labels)["labels"]

    report: dict[str, Any] = {"sample_size": len(articles)}

    if not args.skip_llm:
        print("Scoring with OpenAI (Korean prompt)...", file=sys.stderr)
        t0 = time.time()
        openai_preds = await score_with_openai(articles)
        if openai_preds is None:
            print("  -> SKIPPED: no OPENAI_API_KEY / GPTQ_OPENAI_API_KEY in environment")
            report["openai"] = {"skipped": True, "reason": "no API key in environment"}
        else:
            scoring = compute_accuracy(openai_preds, hand_labels)
            scoring["elapsed_seconds"] = time.time() - t0
            report["openai"] = scoring
            print(f"  -> accuracy: {scoring['accuracy']:.1%} ({scoring['correct']}/{scoring['total']})")

    if not args.skip_finbert:
        print("Scoring with snunlp/KR-FinBert-SC...", file=sys.stderr)
        t0 = time.time()
        finbert_preds = score_with_finbert(articles)
        if finbert_preds is None:
            print("  -> SKIPPED: transformers/torch not available")
            report["finbert"] = {"skipped": True, "reason": "transformers/torch not installed"}
        else:
            scoring = compute_accuracy(finbert_preds, hand_labels)
            scoring["elapsed_seconds"] = time.time() - t0
            report["finbert"] = scoring
            print(f"  -> accuracy: {scoring['accuracy']:.1%} ({scoring['correct']}/{scoring['total']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
