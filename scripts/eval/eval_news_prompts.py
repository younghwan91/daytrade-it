"""Score the news eval set with candidate prompts and measure against outcomes.

Reads data/eval/news_eval_set.jsonl (scripts/eval/build_news_eval_set.py),
asks Claude to judge each (article, ticker) pair under one or more prompt
variants, and reports how well each variant's BUY bucket separates
same-session excess returns.

Evaluation discipline:
- Split by trade day: dev = trade_day <= DEV_END, test = after. Pick
  variants/thresholds on dev only; look at test once for the final choice.
- The bar is NOT zero. Stocks tagged in Toss news beat that day's market
  median on average (attention effect), so a useful BUY bucket has to beat
  the all-articles mean in the same split ("lift").
- LLM outputs are cached per (variant, model, article, ticker), so reruns
  and new scoring rules over the same outputs cost nothing.

Usage:
  uv run python scripts/eval/eval_news_prompts.py --variants v1,v2,v3 --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics as st
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from scipy.stats import spearmanr

EVAL_SET = Path("data/eval/news_eval_set.jsonl")
CACHE_DIR = Path("data/eval/cache")
DEV_END = "2026-09-08"

MATERIAL_TYPES = "수주 / 실적 / 신약 / 증설 / 정책 / 테마 / 루머 / 지분 / 기타"

SYSTEM_V1 = """너는 한국 상장사 기사에서 고정 스키마 필드를 채우는 추출기다.

규칙:
- 주어진 텍스트에 적힌 내용만 쓴다. 배경지식으로 보충하지 않는다.
- 이 작업에 시세 판단은 포함되지 않는다. 스키마에 없는 것은 답하지 않는다.
- 근거가 없으면 가장 보수적인 값(unknown 또는 0)을 고른다.
- 출력은 JSON 객체 하나뿐이다. 코드펜스·설명·주석을 붙이지 않는다."""

USER_V1 = """다음은 종목코드 {code} 관련 기사다.

제목: {title}
본문: {body}
키워드: {category}

아래 7개 필드를 채운 JSON 객체 하나만 출력하라.

material_type: 재료 유형. 다음 중 하나 — {material_types}
persistence: 실적에 반영되는 지속성. one_off(일회성 사건) / multi_quarter(여러 분기에 걸쳐 반영) / structural(사업 구조가 바뀌는 수준)
specificity: 구체성. 0(방향만 언급) / 1(대상과 시기 중 하나만 명시) / 2(둘 다 있으나 금액 없음) / 3(금액·기간·상대방이 모두 명시)
freshness: 정보 신선도. new(처음 알려지는 내용) / follow_up(기존 건의 진행 경과) / repeat(이미 알려진 내용의 재보도)
sentiment_direction: 이 사안이 회사 사업에 미치는 방향. -1(부정) / 0(중립) / 1(긍정)
sentiment_strength: 그 방향의 강도. 0(거의 없음) / 1(작음) / 2(보통) / 3(큼)
scale_vs_size: 회사 규모(시가총액·매출) 대비 사안의 크기. unknown(판단 근거가 본문에 없음) / small / medium / large

출력 형식 예시:
{{"material_type":"수주","persistence":"multi_quarter","specificity":3,"freshness":"new","sentiment_direction":1,"sentiment_strength":2,"scale_vs_size":"medium"}}"""

SYSTEM_V2 = SYSTEM_V1

USER_V2 = """다음은 {name}({code}, {market}, 업종: {sector}) 관련으로 분류된 기사다.

제목: {title}
본문(요약): {body}
분류: {category}

아래 필드를 채운 JSON 객체 하나만 출력하라. 모든 판단은 **{name}** 기준이다.

subject: 이 기사에서 {name}의 비중. main(기사의 주인공) / partial(여러 회사 중 하나로 비중 있게 다룸) / mention(업계 나열·지나가는 언급·다른 회사가 주인공)
reports_price_move: 기사가 {name}의 **이미 일어난** 주가 움직임(급등·급락·상한가·하한가·신고가·장 마감 시세)을 주요 내용으로 보도하는가. true / false
material_type: 재료 유형. 다음 중 하나 — {material_types}
persistence: 실적에 반영되는 지속성. one_off / multi_quarter / structural
specificity: 구체성. 0(방향만) / 1(대상·시기 중 하나) / 2(둘 다, 금액 없음) / 3(금액·기간·상대방 모두)
freshness: new(처음 알려짐) / follow_up(기존 건 진행 경과) / repeat(이미 알려진 내용의 재보도·해설·칼럼)
surprise: 이 내용이 기존 기대와 얼마나 다른가. 0(예상된 일·일정대로) / 1(다소 뜻밖) / 2(크게 뜻밖)
sentiment_direction: {name} 사업에 미치는 방향. -1 / 0 / 1
sentiment_strength: 그 방향의 강도. 0 / 1 / 2 / 3
scale_vs_size: {name} 규모 대비 사안의 크기. unknown / small / medium / large

규칙: 본문에 없는 내용은 추정하지 않는다. 인터뷰·칼럼·업계 동향 기사는 대체로 repeat 이고 강도가 낮다.

출력 형식 예시:
{{"subject":"main","reports_price_move":false,"material_type":"수주","persistence":"multi_quarter","specificity":3,"freshness":"new","surprise":1,"sentiment_direction":1,"sentiment_strength":2,"scale_vs_size":"medium"}}"""

SYSTEM_V3 = """너는 한국 주식 데이트레이더다. 뉴스 요약을 읽고, 이 기사를 본 직후 해당 종목에 들어갔을 때 그날 장 마감까지 시장 대비 어떻게 움직일지 판단한다.

규칙:
- 이미 주가에 반영된 뉴스(급등·상한가를 사후 보도, 재탕 기사)는 추가 상승 여지가 작거나 되돌림이 나올 수 있다는 점을 고려한다.
- 종목이 기사의 주인공이 아니면 반응이 거의 없다.
- 확신이 없으면 0 에 가깝게 답한다.
- 출력은 JSON 객체 하나뿐이다."""

USER_V3 = """종목: {name}({code}, {market}, 업종: {sector})
제목: {title}
본문(요약): {body}
분류: {category}

JSON 으로 답하라.
subject: main / partial / mention
reports_price_move: true / false (이미 일어난 주가 급변을 주로 보도하는가)
expected_reaction: -2(시장 대비 크게 약세) / -1 / 0 / 1 / 2(시장 대비 크게 강세)
conviction: 0(모르겠음) / 1 / 2 / 3(매우 확신)
reason: 한 문장

출력 형식 예시:
{{"subject":"main","reports_price_move":false,"expected_reaction":1,"conviction":2,"reason":"..."}}"""

PROMPTS = {"v1": (SYSTEM_V1, USER_V1), "v2": (SYSTEM_V2, USER_V2), "v3": (SYSTEM_V3, USER_V3)}

PERSIST = {"one_off": 0.0, "multi_quarter": 1.0, "structural": 2.0}
SCALE = {"unknown": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
FRESH = {"repeat": 0.0, "follow_up": 1.0, "new": 2.0}


def extract_json(text: str) -> dict:
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"no json: {text[:100]}")


def v1_production_buy(j: dict) -> float:
    """Replicates the live pipeline: ArticleFeatures.to_sentiment() then
    SignalGenerator.generate_from_news() thresholds. Returns the signal
    confidence if a BUY news signal would be produced, else 0."""
    mag = (
        0.5 * PERSIST[j["persistence"]]
        + 0.3 * SCALE[j["scale_vs_size"]]
        + 0.2 * int(j["specificity"])
    ) / 2.5
    score = int(j["sentiment_direction"]) * (int(j["sentiment_strength"]) / 3) * (0.5 + 0.5 * mag)
    conf = 0.5 * int(j["specificity"]) / 3 + 0.5 * FRESH[j["freshness"]] / 2
    if score <= 0.15 or conf < 0.5:
        return 0.0
    sig_conf = conf * 0.7 + abs(score) * 0.3
    return sig_conf if sig_conf >= 0.5 else 0.0


def v2_score(j: dict, *, drop_mention: bool, drop_price_move: bool, use_surprise: bool) -> float:
    if drop_mention and j.get("subject") == "mention":
        return 0.0
    if drop_price_move and j.get("reports_price_move") in (True, "true"):
        return 0.0
    mag = (
        0.5 * PERSIST[j["persistence"]]
        + 0.3 * SCALE[j["scale_vs_size"]]
        + 0.2 * int(j["specificity"])
    ) / 2.5
    s = int(j["sentiment_direction"]) * (int(j["sentiment_strength"]) / 3) * (0.5 + 0.5 * mag)
    if use_surprise:
        s *= 0.5 + 0.25 * int(j.get("surprise", 0))
    if j.get("freshness") == "repeat":
        s *= 0.5
    return s


def v3_score(j: dict, *, drop_mention: bool, drop_price_move: bool) -> float:
    if drop_mention and j.get("subject") == "mention":
        return 0.0
    if drop_price_move and j.get("reports_price_move") in (True, "true"):
        return 0.0
    return int(j["expected_reaction"]) * (int(j["conviction"]) / 3)


RULES: dict[str, list[tuple[str, Callable[[dict], float], float]]] = {
    "v1": [("v1/pipeline", v1_production_buy, 0.0)],
    "v2": [
        (
            "v2/raw",
            lambda j: v2_score(j, drop_mention=False, drop_price_move=False, use_surprise=False),
            0.15,
        ),
        (
            "v2/-mention",
            lambda j: v2_score(j, drop_mention=True, drop_price_move=False, use_surprise=False),
            0.15,
        ),
        (
            "v2/-mention-pricemove",
            lambda j: v2_score(j, drop_mention=True, drop_price_move=True, use_surprise=False),
            0.15,
        ),
        (
            "v2/-mention-pricemove+surprise",
            lambda j: v2_score(j, drop_mention=True, drop_price_move=True, use_surprise=True),
            0.1,
        ),
    ],
    "v3": [
        ("v3/raw", lambda j: v3_score(j, drop_mention=False, drop_price_move=False), 0.3),
        (
            "v3/-mention-pricemove",
            lambda j: v3_score(j, drop_mention=True, drop_price_move=True),
            0.3,
        ),
    ],
}


@dataclass
class Scored:
    row: dict
    judgment: dict


async def judge_all(rows: list[dict], variant: str, model: str) -> list[Scored]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{variant}__{model}.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.open():
            rec = json.loads(line)
            cache[rec["key"]] = rec["judgment"]

    system, user_tpl = PROMPTS[variant]
    sem = asyncio.Semaphore(6)
    key_of = lambda r: f"{r['article_id']}|{r['ticker']}"  # noqa: E731

    async with httpx.AsyncClient(timeout=60) as client:

        async def one(r: dict) -> None:
            key = key_of(r)
            if key in cache:
                return
            user = user_tpl.format(
                code=r["ticker"],
                name=r["name"],
                market=r["market"],
                sector=r["sector"] or "-",
                title=r["title"],
                body=r["summary"] or "",
                category=r["category"] or "",
                material_types=MATERIAL_TYPES,
            )
            async with sem:
                for attempt in range(3):
                    try:
                        resp = await client.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json",
                            },
                            json={
                                "model": model,
                                "system": system,
                                "temperature": 0,
                                "max_tokens": 400,
                                "messages": [{"role": "user", "content": user}],
                            },
                        )
                        resp.raise_for_status()
                        text = "".join(
                            b.get("text", "") for b in resp.json()["content"] if b["type"] == "text"
                        )
                        cache[key] = extract_json(text)
                        with cache_path.open("a") as f:
                            f.write(
                                json.dumps({"key": key, "judgment": cache[key]}, ensure_ascii=False)
                                + "\n"
                            )
                        return
                    except Exception as exc:  # noqa: BLE001
                        if attempt == 2:
                            print(f"FAILED {variant} {key}: {exc}")
                        await asyncio.sleep(2 * (attempt + 1))

        await asyncio.gather(*(one(r) for r in rows))

    return [Scored(r, cache[key_of(r)]) for r in rows if key_of(r) in cache]


def boot_ci(vals: list[float], n: int = 2000) -> tuple[float, float]:
    rnd = random.Random(0)
    bs = sorted(st.mean(rnd.choices(vals, k=len(vals))) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]


def report(label: str, scored: list[Scored], fn: Callable[[dict], float], thr: float) -> None:
    for split, keep in (
        ("dev", lambda r: r["trade_day"] <= DEV_END),
        ("test", lambda r: r["trade_day"] > DEV_END),
    ):
        s = [x for x in scored if keep(x.row)]
        if not s:
            continue
        scores, ex = [], []
        for x in s:
            try:
                scores.append(fn(x.judgment))
                ex.append(x.row["excess"])
            except (KeyError, ValueError, TypeError):
                continue
        base = st.mean(ex)
        buy = [e for sc, e in zip(scores, ex, strict=True) if sc > thr]
        rest = [e for sc, e in zip(scores, ex, strict=True) if sc <= thr]
        rho = spearmanr(scores, ex).statistic if len(set(scores)) > 1 else float("nan")
        if buy:
            lo, hi = boot_ci(buy)
            hit = sum(e > 0 for e in buy) / len(buy)
            print(
                f"  {label:34s} {split:4s} n={len(ex):3d} base={base:+.4f} | BUY n={len(buy):3d} "
                f"mean={st.mean(buy):+.4f} [{lo:+.4f},{hi:+.4f}] lift={st.mean(buy) - base:+.4f} "
                f"hit={hit:.0%} | rest={st.mean(rest) if rest else float('nan'):+.4f} | rho={rho:+.3f}"
            )
        else:
            print(
                f"  {label:34s} {split:4s} n={len(ex):3d} base={base:+.4f} | BUY n=0 | rho={rho:+.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="v1,v2,v3")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--show-test", action="store_true", help="also print the held-out test split"
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in EVAL_SET.open()]
    global DEV_END  # noqa: PLW0603
    for variant in args.variants.split(","):
        scored = asyncio.run(judge_all(rows, variant, args.model))
        print(f"\n[{variant} / {args.model}] judged {len(scored)}/{len(rows)}")
        for label, fn, thr in RULES[variant]:
            if args.show_test:
                report(label, scored, fn, thr)
            else:
                report(label, [x for x in scored if x.row["trade_day"] <= DEV_END], fn, thr)


if __name__ == "__main__":
    main()
