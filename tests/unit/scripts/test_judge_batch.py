"""judge_batch (plan C4): Batch API over httpx.MockTransport -- never the network."""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from judge_batch import (  # noqa: E402
    BatchJudge,
    BudgetExceeded,
    build_request,
    custom_id_of,
    estimate_cost,
    stratified_sample,
)

from daytrade_it.infrastructure.external.anthropic_client import (
    _MAX_OUTPUT_TOKENS,
    ArticleFeatures,
    build_prompt,
)

FEATS = {
    "subject": "main",
    "reports_price_move": False,
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "surprise": 1,
    "sentiment_direction": 1,
    "sentiment_strength": 2,
    "scale_vs_size": "medium",
}
UUID = "595eead7-3889-57cd-bdc5-fc299f660c70"


def _pair(i: int, month: str = "2026-09", sector: str = "IT", session: str = "intraday") -> dict:
    return {
        "key": f"{UUID[:-4]}{i:04d}|{i:06d}",
        "code": f"{i:06d}",
        "name": f"회사{i}",
        "market": "KOSPI",
        "sector": sector,
        "title": f"제목 {i}",
        "body": "요약 " * 50,
        "category": None,
        "month": month,
        "session": session,
    }


def test_build_request_uses_live_prompt_and_limits() -> None:
    p = _pair(1)
    req = build_request(p, model="claude-haiku-4-5")
    system, user = build_prompt(
        code=p["code"],
        name=p["name"],
        market=p["market"],
        sector=p["sector"],
        title=p["title"],
        body=p["body"],
        category=p["category"],
    )
    assert req["custom_id"] == custom_id_of(p["key"])
    assert len(req["custom_id"]) <= 64 and "|" not in req["custom_id"]
    assert req["params"] == {
        "model": "claude-haiku-4-5",
        "system": system,
        "temperature": 0,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": user}],
    }


def test_stratified_sample_is_proportional_seeded_and_covers_strata() -> None:
    pairs = [_pair(i, month="2026-08" if i < 80 else "2026-09") for i in range(100)]
    a = stratified_sample(pairs, 10, ["month", "sector", "session"], seed=7)
    b = stratified_sample(list(reversed(pairs)), 10, ["month", "sector", "session"], seed=7)
    assert [p["key"] for p in a] == [p["key"] for p in b]
    months = [p["month"] for p in a]
    assert months.count("2026-08") == 8 and months.count("2026-09") == 2
    assert stratified_sample(pairs, 500, ["month"], seed=1) == sorted(pairs, key=lambda p: p["key"])


def test_estimate_cost_is_batch_discounted_haiku_price() -> None:
    reqs = [build_request(_pair(i), model="m") for i in range(10)]
    est = estimate_cost(reqs, chars_per_token=1.0, output_tokens=200)
    chars = sum(
        len(r["params"]["system"]) + len(r["params"]["messages"][0]["content"]) for r in reqs
    )
    assert est["input_tokens"] == chars
    assert est["usd"] == pytest.approx((chars * 1.0 + 10 * 200 * 5.0) / 1e6 * 0.5)


class FakeAPI:
    """Minimal Message Batches API: create -> in_progress once -> ended."""

    def __init__(self, result_for=None) -> None:
        self.created: list[dict] = []
        self.polls = 0
        self.result_for = result_for or (lambda _cid: {"type": "succeeded", "message": _msg(FEATS)})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "k"
        assert request.headers["anthropic-version"] == "2023-06-01"
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            body = json.loads(request.content)
            self.created.append(body)
            return httpx.Response(
                200, json={"id": f"b{len(self.created)}", "processing_status": "in_progress"}
            )
        if (
            request.method == "GET"
            and path.startswith("/v1/messages/batches/")
            and not path.endswith("/results")
        ):
            bid = path.rsplit("/", 1)[1]
            self.polls += 1
            status = "ended" if self.polls % 2 == 0 else "in_progress"
            return httpx.Response(
                200,
                json={
                    "id": bid,
                    "processing_status": status,
                    "results_url": f"https://api.anthropic.com/v1/messages/batches/{bid}/results"
                    if status == "ended"
                    else None,
                },
            )
        if path.endswith("/results"):
            bid = path.split("/")[-2]
            reqs = self.created[int(bid[1:]) - 1]["requests"]
            lines = [
                json.dumps({"custom_id": r["custom_id"], "result": self.result_for(r["custom_id"])})
                for r in reqs
            ]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return httpx.Response(404)


def _msg(payload: dict | str) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1300, "output_tokens": 150},
    }


def _judge(tmp_path: Path, api: FakeAPI, budget: float = 100.0) -> BatchJudge:
    client = httpx.Client(transport=httpx.MockTransport(api), base_url="https://api.anthropic.com")
    return BatchJudge(
        client=client,
        api_key="k",
        model="claude-haiku-4-5",
        cache=tmp_path / "cache.jsonl",
        errors=tmp_path / "errors.jsonl",
        state=tmp_path / "state.json",
        budget_usd=budget,
        sleep=lambda _s: None,
    )


def test_budget_abort_submits_nothing(tmp_path: Path) -> None:
    api = FakeAPI()
    j = _judge(tmp_path, api, budget=0.0001)
    with pytest.raises(BudgetExceeded):
        j.run([_pair(i) for i in range(5)], sample=5, stratify=["month"], seed=1)
    assert api.created == []
    assert not (tmp_path / "state.json").exists()


def test_run_submits_polls_and_writes_cache_skipping_cached(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(4)]
    (tmp_path / "cache.jsonl").write_text(
        json.dumps({"key": pairs[0]["key"], "v2": {"score": 0}}) + "\n"
    )
    api = FakeAPI()
    j = _judge(tmp_path, api)
    summary = j.run(pairs, sample=100, stratify=["month"], seed=1)
    assert len(api.created) == 1
    assert len(api.created[0]["requests"]) == 3
    rows = [json.loads(line) for line in (tmp_path / "cache.jsonl").open()]
    assert [r["key"] for r in rows[1:]] == sorted(p["key"] for p in pairs[1:])
    assert rows[1]["v2"]["score"] == pytest.approx(ArticleFeatures(**FEATS).score())
    assert set(rows[1]["v2"]) == set(FEATS) | {"score"}
    assert summary["succeeded"] == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert all(b["processed"] for b in state["batches"])


def test_results_errored_and_schema_failures_go_to_errors_file(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(3)]
    cids = [custom_id_of(p["key"]) for p in sorted(pairs, key=lambda p: p["key"])]

    def result_for(cid: str) -> dict:
        if cid == cids[0]:
            return {
                "type": "errored",
                "error": {"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
            }
        if cid == cids[1]:
            return {"type": "succeeded", "message": _msg({**FEATS, "persistence": "forever"})}
        return {"type": "succeeded", "message": _msg(FEATS)}

    api = FakeAPI(result_for)
    summary = _judge(tmp_path, api).run(pairs, sample=10, stratify=["month"], seed=1)
    assert (summary["succeeded"], summary["failed"]) == (1, 2)
    errs = [json.loads(line) for line in (tmp_path / "errors.jsonl").open()]
    assert {e["reason"] for e in errs} == {"errored", "schema"}
    # one retry allowed: a second run resubmits both; a third run submits nothing
    _judge(tmp_path, FakeAPI(result_for)).run(pairs, sample=10, stratify=["month"], seed=1)
    api3 = FakeAPI(result_for)
    _judge(tmp_path, api3).run(pairs, sample=10, stratify=["month"], seed=1)
    assert api3.created == []


def test_resume_polls_pending_batch_without_resubmitting(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(2)]
    api = FakeAPI()
    j = _judge(tmp_path, api)
    # simulate a crash right after submission
    reqs = [build_request(p, model="claude-haiku-4-5") for p in pairs]
    j.submit(reqs, {r["custom_id"]: p["key"] for r, p in zip(reqs, pairs, strict=True)})
    assert len(api.created) == 1
    summary = _judge(tmp_path, api).run(pairs + [_pair(9)], sample=10, stratify=["month"], seed=1)
    assert len(api.created) == 1  # resumed, nothing new submitted
    assert summary["resumed"] == 1 and summary["succeeded"] == 2
    keys = {json.loads(line)["key"] for line in (tmp_path / "cache.jsonl").open()}
    assert keys == {p["key"] for p in pairs}


def test_dry_run_makes_no_http_calls(tmp_path: Path) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry run must not call the API")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    j = BatchJudge(
        client=client,
        api_key=None,
        model="claude-haiku-4-5",
        cache=tmp_path / "c.jsonl",
        errors=tmp_path / "e.jsonl",
        state=tmp_path / "s.json",
        budget_usd=0.0,
        sleep=lambda _s: None,
    )
    summary = j.run(
        [_pair(i) for i in range(20)],
        sample=5,
        stratify=["month", "sector", "session"],
        seed=3,
        dry_run=True,
    )
    assert summary["sample"] == 5
    assert summary["estimate"]["usd"] > 0
