"""Judge historical (article, code) pairs with the Message Batches API (A-plan contract C4).

    # cost estimate only, no API call
    uv run python scripts/eval/judge_batch.py --pairs data/eval/universe_pairs.jsonl \\
        --sample 10000 --stratify month,sector,session --seed 1 --budget-usd 15 --dry-run
    # submit, poll, write the cache (resumable: rerun the same command after a crash)
    uv run python scripts/eval/judge_batch.py --pairs ... --sample 10000 --seed 1 --budget-usd 15

- Prompt = the live `anthropic_client.build_prompt`, request params = the live
  `_complete` call (temperature 0, `_MAX_OUTPUT_TOKENS`), parsing = the live
  `parse_features` (fixed-schema validation). Nothing is copied.
- Pairs already in the cache (`data/eval/cache/universe_v2.jsonl`, rows
  `{"key", "input_hash", "v2"}` with v2 = asdict(features) + "score") are
  skipped only if the row's input_hash = sha256 of (model, system, user) of the
  request this pair would send now (`cache_row_valid`; legacy rows without a
  hash count only if the pair's input is unchanged by the body fallback); so are
  pairs that already failed in `MAX_ATTEMPTS` (3) distinct batches (errors file).
- Stratified sample: proportional allocation over the `--stratify` fields
  (largest remainder), seeded shuffle within each stratum -- deterministic for
  a given (pairs set, sample, seed) regardless of file order.
- Budget: before submitting, estimate cost = input tokens (prompt chars /
  `--chars-per-token`, default 1.0 -- measured ~1.3 chars/token on the live
  prompt, so ~30% conservative) at $1/M + `--est-output-tokens` (default 200;
  measured ~150) at $5/M, times the 50% Batch discount. Abort if over
  `--budget-usd`. `--calibrate N` measures chars/token on N sampled requests
  with the free count_tokens endpoint first (not in --dry-run).
- A batch create is never retried: a 5xx/transport error records the attempt
  and fails loudly (`SubmitUncertain`); a rerun resubmits that chunk only if no
  unrecorded batch of that size is listed, else `--adopt-batch ID` records it.
- Resumable, no double spend: the chosen sample (keys, chunking, seed, args)
  is saved to the state file before the first submit; each batch id is appended
  to `<state>.batches.log` as soon as the create call returns, then recorded in
  the state. A rerun of an incomplete run submits only chunks without a batch
  and collects unprocessed batches (poll with backoff, results_url JSONL). A
  completed run blocks drawing a fresh sample unless `--new-run`.
- Raw httpx (no `anthropic` SDK in this project), key from ANTHROPIC_API_KEY or
  ../quant-airflow/.env like the live daemon.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import httpx

from daytrade_it.application.news.universe import judgment_input_hash
from daytrade_it.infrastructure.external.anthropic_client import (
    _MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    SchemaError,
    build_prompt,
    parse_features,
)

API = "https://api.anthropic.com"
CACHE = Path("data/eval/cache/universe_v2.jsonl")
ERRORS = Path("data/eval/cache/universe_v2_errors.jsonl")
STATE = Path("data/eval/cache/universe_batch_state.json")

INPUT_USD_PER_M = 1.0  # Haiku 4.5
OUTPUT_USD_PER_M = 5.0
BATCH_DISCOUNT = 0.5
MAX_ATTEMPTS = 3  # same as live score_article (_MAX_RETRIES + 1)
CHUNK = 10_000  # requests per batch (API limit 100k / 256 MB)


class BudgetExceeded(RuntimeError):
    pass


class SubmitUncertain(RuntimeError):
    """A batch create may or may not have happened (5xx / transport error): never retried."""


#: How far before a failed create a listed batch may claim to be created (clock skew).
UNCERTAIN_SKEW = dt.timedelta(minutes=10)


def custom_id_of(key: str) -> str:
    """`uuid|code` -> API-safe custom_id (^[a-zA-Z0-9_-]{1,64}$)."""
    return key.replace("|", "_")


def build_request(pair: dict, model: str) -> dict:
    system, user = build_prompt(
        code=pair["code"],
        name=pair["name"],
        market=pair.get("market"),
        sector=pair.get("sector"),
        title=pair["title"],
        body=pair.get("body") or "",
        category=pair.get("category"),
    )
    return {
        "custom_id": custom_id_of(pair["key"]),
        "params": {
            "model": model,
            "system": system,
            "temperature": 0,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": user}],
        },
    }


def request_input_hash(req: dict) -> str:
    p = req["params"]
    return judgment_input_hash(p["model"], p["system"], p["messages"][0]["content"])


def pair_input_hash(pair: dict, model: str) -> str:
    """Hash of the exact input `build_request` would send for this pair."""
    return request_input_hash(build_request(pair, model))


def cache_row_valid(row: dict, pair: dict, model: str) -> bool:
    """Whether a cache row judged this pair's current input.

    Rows carry `input_hash`. Legacy rows (written before hashing) are valid only
    if the pair's current input hash equals the hash of the input the old
    pipeline sent -- body = the raw C1 summary (no title fallback) -- i.e. the
    pair's input didn't change.
    """
    current = pair_input_hash(pair, model)
    if "input_hash" in row:
        return row["input_hash"] == current
    legacy = {**pair, "body": pair["summary"]} if "summary" in pair else pair
    return pair_input_hash(legacy, model) == current


def _chars(req: dict) -> int:
    p = req["params"]
    return len(p["system"]) + sum(len(m["content"]) for m in p["messages"])


def estimate_cost(requests: list[dict], chars_per_token: float, output_tokens: int) -> dict:
    input_tokens = round(sum(_chars(r) for r in requests) / chars_per_token)
    out = len(requests) * output_tokens
    usd = (input_tokens * INPUT_USD_PER_M + out * OUTPUT_USD_PER_M) / 1e6 * BATCH_DISCOUNT
    return {
        "requests": len(requests),
        "input_tokens": input_tokens,
        "output_tokens": out,
        "usd": usd,
    }


def _stratum(pair: dict, fields: list[str]) -> tuple[str, ...]:
    return tuple(str(pair.get(f)) for f in fields)


def stratified_sample(pairs: list[dict], n: int, fields: list[str], seed: int) -> list[dict]:
    """Proportional stratified sample, sorted by key. All pairs if n >= len(pairs)."""
    ordered = sorted(pairs, key=lambda p: p["key"])
    if n >= len(ordered):
        return ordered
    strata: dict[tuple[str, ...], list[dict]] = {}
    for p in ordered:
        strata.setdefault(_stratum(p, fields), []).append(p)
    total = len(ordered)
    quotas = {s: n * len(ps) / total for s, ps in strata.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    rest = n - sum(alloc.values())
    for s in sorted(strata, key=lambda s: (-(quotas[s] - alloc[s]), s))[:rest]:
        alloc[s] += 1
    rng = random.Random(seed)
    picked = []
    for s in sorted(strata):
        picked += rng.sample(strata[s], alloc[s])
    return sorted(picked, key=lambda p: p["key"])


class BatchJudge:
    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str | None,
        model: str,
        cache: Path,
        errors: Path,
        state: Path,
        budget_usd: float,
        chars_per_token: float = 1.0,
        est_output_tokens: int = 200,
        chunk: int = CHUNK,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.api_key = api_key
        self.model = model
        self.cache, self.errors, self.state = cache, errors, state
        self.budget_usd = budget_usd
        self.chars_per_token = chars_per_token
        self.est_output_tokens = est_output_tokens
        self.chunk = chunk
        self.sleep = sleep
        self.log = log

    # --- local files ----------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state.exists():
            return json.loads(self.state.read_text())
        return {"batches": []}

    def _save_state(self, state: dict) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state)

    def cached_rows(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        if self.cache.exists():
            for line in self.cache.open():
                if line.strip():
                    row = json.loads(line)
                    out.setdefault(row["key"], []).append(row)
        return out

    def is_cached(self, pair: dict, rows: dict[str, list[dict]]) -> bool:
        return any(cache_row_valid(r, pair, self.model) for r in rows.get(pair["key"], []))

    def failure_counts(self) -> Counter[str]:
        if not self.errors.exists():
            return Counter()
        seen = {
            (r["key"], r.get("batch_id"))
            for line in self.errors.open()
            if line.strip() and (r := json.loads(line))
        }
        return Counter(key for key, _ in seen)

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- HTTP -----------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _http(self, method: str, url: str, **kw) -> httpx.Response:
        delay = 5.0
        for attempt in range(6):
            resp = self.client.request(method, url, headers=self._headers(), timeout=120.0, **kw)
            if resp.status_code not in (429, 500, 502, 503, 529) or attempt == 5:
                resp.raise_for_status()
                return resp
            self.log(f"HTTP {resp.status_code} on {method} {url} -- retry in {delay:.0f}s")
            self.sleep(delay)
            delay = min(delay * 2, 300.0)
        raise AssertionError("unreachable")

    def count_tokens(self, req: dict) -> int:
        p = req["params"]
        body = {"model": p["model"], "system": p["system"], "messages": p["messages"]}
        return int(
            self._http("POST", f"{API}/v1/messages/count_tokens", json=body).json()["input_tokens"]
        )

    @property
    def batches_log(self) -> Path:
        return self.state.with_suffix(".batches.log")

    def submit(self, requests: list[dict], keys: dict[str, str], chunk: int) -> str:
        """Create one batch. Exactly one POST: a create is billable, so it is never retried.

        A 5xx or transport error leaves it unknown whether the batch exists: the
        attempt is recorded in the state (`uncertain_submits`) and in the
        batches log, and `SubmitUncertain` is raised. A rerun lists recent
        batches before submitting that chunk again (`_resolve_uncertain`).
        """
        attempted_at = dt.datetime.now(dt.UTC).isoformat()
        custom_ids = {r["custom_id"]: keys[r["custom_id"]] for r in requests}
        hashes = {r["custom_id"]: request_input_hash(r) for r in requests}
        try:
            resp = self.client.request(
                "POST",
                f"{API}/v1/messages/batches",
                headers=self._headers(),
                timeout=120.0,
                json={"requests": requests},
            )
        except httpx.TransportError as exc:
            self._uncertain(chunk, custom_ids, hashes, attempted_at, f"{type(exc).__name__}: {exc}")
            raise AssertionError("unreachable") from exc
        if resp.status_code >= 500:
            self._uncertain(chunk, custom_ids, hashes, attempted_at, f"HTTP {resp.status_code}")
        resp.raise_for_status()  # 4xx (incl. 429): rejected, nothing was created
        batch = resp.json()
        self._record_batch(batch["id"], chunk, custom_ids, hashes)
        return batch["id"]

    def _record_batch(
        self, batch_id: str, chunk: int, custom_ids: dict[str, str], hashes: dict[str, str]
    ) -> None:
        now = dt.datetime.now(dt.UTC).isoformat()
        # the batch is billable from here: log its id before anything else can fail
        self._append(
            self.batches_log,
            {"id": batch_id, "chunk": chunk, "requests": len(custom_ids), "submitted_at": now},
        )
        state = self._load_state()
        state["batches"].append(
            {
                "id": batch_id,
                "chunk": chunk,
                "custom_ids": custom_ids,
                "input_hashes": hashes,
                "submitted_at": now,
                "processed": False,
            }
        )
        pending = [u for u in state.get("uncertain_submits", []) if u["chunk"] != chunk]
        if pending:
            state["uncertain_submits"] = pending
        else:
            state.pop("uncertain_submits", None)
        self._save_state(state)
        self.log(f"submitted batch {batch_id} chunk {chunk} ({len(custom_ids)} requests)")

    def _uncertain(
        self,
        chunk: int,
        custom_ids: dict[str, str],
        hashes: dict[str, str],
        attempted_at: str,
        error: str,
    ) -> None:
        entry = {
            "chunk": chunk,
            "requests": len(custom_ids),
            "attempted_at": attempted_at,
            "error": error,
            "custom_ids": custom_ids,
            "input_hashes": hashes,
        }
        self._append(
            self.batches_log,
            {
                "uncertain": True,
                **{k: v for k, v in entry.items() if k not in ("custom_ids", "input_hashes")},
            },
        )
        state = self._load_state()
        state["uncertain_submits"] = [
            u for u in state.get("uncertain_submits", []) if u["chunk"] != chunk
        ] + [entry]
        self._save_state(state)
        candidates = self._candidates(entry)
        raise SubmitUncertain(self._uncertain_message(entry, candidates))

    def _candidates(self, entry: dict) -> list[str] | None:
        """Unknown recent batches that could be this create (None if listing failed)."""
        try:
            listed = self._http("GET", f"{API}/v1/messages/batches", params={"limit": 100}).json()
        except (httpx.HTTPError, ValueError) as exc:
            self.log(f"listing batches failed: {exc}")
            return None
        known = {b["id"] for b in self._load_state().get("batches", [])}
        since = dt.datetime.fromisoformat(entry["attempted_at"]) - UNCERTAIN_SKEW
        out = []
        for b in listed.get("data") or []:
            try:
                created = dt.datetime.fromisoformat(str(b["created_at"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                created = None
            total = sum((b.get("request_counts") or {}).values())
            if b.get("id") in known or (created is not None and created < since):
                continue
            if total == entry["requests"]:
                out.append(b["id"])
        return out

    def _uncertain_message(self, entry: dict, candidates: list[str] | None) -> str:
        if candidates is None:
            found = "could not list batches"
        elif candidates:
            found = f"unrecorded batch(es) that may be it: {', '.join(candidates)}"
        else:
            found = "no matching batch listed"
        return (
            f"batch create for chunk {entry['chunk']} ({entry['requests']} requests) at "
            f"{entry['attempted_at']} failed with {entry['error']} and was NOT retried; {found}. "
            f"Check GET /v1/messages/batches and {self.batches_log}. If a batch was created, "
            "record it with --adopt-batch ID; otherwise rerun (it resubmits only if no "
            "matching batch is listed)."
        )

    def _resolve_uncertain(self, chunk: int) -> None:
        """Before resubmitting a chunk whose create was uncertain: refuse if a batch may exist."""
        entry = next(
            (u for u in self._load_state().get("uncertain_submits", []) if u["chunk"] == chunk),
            None,
        )
        if entry is None:
            return
        candidates = self._candidates(entry)
        if candidates is None or candidates:
            raise SubmitUncertain(self._uncertain_message(entry, candidates))
        self.log(f"chunk {chunk}: no batch from the failed create is listed -- resubmitting")

    def adopt_batch(self, batch_id: str) -> None:
        """Record an existing batch for the uncertain create it belongs to (matched by size)."""
        batch = self._http("GET", f"{API}/v1/messages/batches/{batch_id}").json()
        total = sum((batch.get("request_counts") or {}).values())
        state = self._load_state()
        if batch_id in {b["id"] for b in state.get("batches", [])}:
            raise ValueError(f"{batch_id} is already recorded")
        matches = [u for u in state.get("uncertain_submits", []) if u["requests"] == total]
        if len(matches) != 1:
            raise ValueError(
                f"{batch_id} has {total} requests: {len(matches)} uncertain create(s) match"
            )
        u = matches[0]
        self._record_batch(batch_id, u["chunk"], u["custom_ids"], u["input_hashes"])

    def wait(self, batch_id: str, first: float = 30.0, cap: float = 600.0) -> dict:
        delay = first
        while True:
            batch = self._http("GET", f"{API}/v1/messages/batches/{batch_id}").json()
            if batch["processing_status"] == "ended":
                return batch
            self.log(
                f"batch {batch_id}: {batch['processing_status']} {batch.get('request_counts', '')}"
            )
            self.sleep(delay)
            delay = min(delay * 1.5, cap)

    def collect(self, entry: dict) -> tuple[int, int]:
        """Poll one state entry to completion and write its results. -> (succeeded, failed)."""
        batch = self.wait(entry["id"])
        text = self._http("GET", batch["results_url"]).text
        keys = entry["custom_ids"]
        hashes = entry.get("input_hashes") or {}
        ok = bad = 0
        seen = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = keys.get(row["custom_id"])
            if key is None:
                continue
            seen.add(row["custom_id"])
            result = row.get("result") or {}
            if result.get("type") != "succeeded":
                self._append(
                    self.errors,
                    {
                        "key": key,
                        "reason": result.get("type") or "unknown",
                        "detail": result.get("error"),
                        "batch_id": entry["id"],
                    },
                )
                bad += 1
                continue
            blocks = (result.get("message") or {}).get("content") or []
            content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            try:
                features = parse_features(content)
            except SchemaError as exc:
                self._append(
                    self.errors,
                    {"key": key, "reason": "schema", "detail": str(exc), "batch_id": entry["id"]},
                )
                bad += 1
                continue
            cached: dict = {"key": key}
            # the hash of the input actually submitted (entries from before hashing have none)
            if (h := hashes.get(row["custom_id"])) is not None:
                cached["input_hash"] = h
            cached["v2"] = {**asdict(features), "score": features.score()}
            self._append(self.cache, cached)
            ok += 1
        for cid in set(keys) - seen:
            self._append(
                self.errors,
                {
                    "key": keys[cid],
                    "reason": "missing_result",
                    "detail": None,
                    "batch_id": entry["id"],
                },
            )
            bad += 1
        state = self._load_state()
        for b in state["batches"]:
            if b["id"] == entry["id"]:
                b["processed"] = True
        self._save_state(state)
        self.log(f"batch {entry['id']}: {ok} cached, {bad} failed")
        return ok, bad

    # --- orchestration ------------------------------------------------------------

    def run(
        self,
        pairs: list[dict],
        *,
        sample: int,
        stratify: list[str],
        seed: int,
        dry_run: bool = False,
        calibrate: int = 0,
        new_run: bool = False,
    ) -> dict:
        """Draw + persist a sample, submit its chunks, collect. Reruns continue the stored run.

        State `run` holds the chosen keys and chunking (written before the first
        submit). A rerun with an incomplete run submits only chunks that have no
        batch yet and collects unprocessed batches -- the CLI's sample/seed are
        ignored then. A completed run blocks a fresh sample unless `new_run`.
        """
        summary: dict = {"resumed": 0, "submitted": 0, "succeeded": 0, "failed": 0}
        state = self._load_state()
        run = state.get("run")
        if run and not run.get("completed"):
            if dry_run:
                self.log(f"incomplete run in {self.state} -- a real run resumes it")
                summary["pending_run"] = True
                return summary
            summary["resumed"] = 1
            self.log(f"resuming stored run (seed {run['seed']}, {len(run['keys'])} keys)")
            return self._finish(run, {p["key"]: p for p in pairs}, summary)
        if run and not new_run:
            self.log(f"completed run in {self.state}; pass --new-run to draw a fresh sample")
            summary["refused"] = True
            return summary

        cached = self.cached_rows()
        failures = self.failure_counts()
        unique = {p["key"]: p for p in pairs}
        todo = [
            p
            for k, p in unique.items()
            if failures[k] < MAX_ATTEMPTS and not self.is_cached(p, cached)
        ]
        chosen = stratified_sample(todo, sample, stratify, seed)
        requests = [build_request(p, self.model) for p in chosen]
        if calibrate and requests and not dry_run:
            probe = random.Random(seed).sample(requests, min(calibrate, len(requests)))
            ratio = sum(_chars(r) for r in probe) / sum(self.count_tokens(r) for r in probe)
            self.chars_per_token = ratio / 1.1  # 10% safety margin
            self.log(f"calibrated chars/token {ratio:.3f} -> using {self.chars_per_token:.3f}")
        est = estimate_cost(requests, self.chars_per_token, self.est_output_tokens)
        summary.update(
            {
                "pairs": len(unique),
                "cached": len(unique) - len(todo),
                "eligible": len(todo),
                "sample": len(requests),
                "estimate": est,
            }
        )
        self.log(
            f"pairs {len(unique)}, eligible {len(todo)}, sample {len(requests)}; "
            f"estimate {est['input_tokens']:,} in + {est['output_tokens']:,} out tokens "
            f"= ${est['usd']:.2f} (batch-discounted) vs budget ${self.budget_usd:.2f}"
        )
        if dry_run:
            summary["over_budget"] = est["usd"] > self.budget_usd
            return summary
        if est["usd"] > self.budget_usd:
            raise BudgetExceeded(f"estimate ${est['usd']:.2f} > budget ${self.budget_usd:.2f}")
        if not chosen:
            self.log("nothing to judge")
            return summary
        keys = [p["key"] for p in chosen]
        run = {
            "seed": seed,
            "sample": sample,
            "stratify": stratify,
            "model": self.model,
            "estimate": est,
            "budget_usd": self.budget_usd,
            "keys": keys,
            "chunks": [keys[i : i + self.chunk] for i in range(0, len(keys), self.chunk)],
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "completed": False,
        }
        history = state.get("history", [])
        if state.get("run"):
            history.append({"run": state["run"], "batches": state["batches"]})
        self._save_state({"run": run, "batches": [], "history": history})
        return self._finish(run, unique, summary)

    def _finish(self, run: dict, by_key: dict[str, dict], summary: dict) -> dict:
        submitted = {b["chunk"] for b in self._load_state()["batches"]}
        remaining: list[tuple[int, list[str], list[dict]]] = []
        for i, chunk_keys in enumerate(run["chunks"]):
            if i in submitted:
                continue
            missing = [k for k in chunk_keys if k not in by_key]
            if missing:
                raise KeyError(
                    f"{len(missing)} stored sample key(s) not in --pairs, e.g. {missing[0]}"
                )
            reqs = [build_request(by_key[k], run["model"]) for k in chunk_keys]
            remaining.append((i, chunk_keys, reqs))
        if remaining:
            est = estimate_cost(
                [r for *_, reqs in remaining for r in reqs],
                self.chars_per_token,
                self.est_output_tokens,
            )
            self.log(
                f"run original estimate ${run['estimate']['usd']:.2f} under budget "
                f"${run['budget_usd']:.2f}; remaining {len(remaining)}/{len(run['chunks'])} "
                f"chunk(s) estimate ${est['usd']:.2f} vs current budget ${self.budget_usd:.2f}"
            )
            summary["remaining_estimate"] = est
            # a resume spends real money too: check the unsubmitted part against today's budget
            if est["usd"] > self.budget_usd:
                raise BudgetExceeded(
                    f"remaining estimate ${est['usd']:.2f} > budget ${self.budget_usd:.2f}"
                )
        for i, chunk_keys, reqs in remaining:
            self._resolve_uncertain(i)
            self.submit(reqs, {custom_id_of(k): k for k in chunk_keys}, i)
            summary["submitted"] += 1
        for entry in [b for b in self._load_state()["batches"] if not b["processed"]]:
            ok, bad = self.collect(entry)
            summary["succeeded"] += ok
            summary["failed"] += bad
        state = self._load_state()
        state["run"]["completed"] = True
        self._save_state(state)
        return summary


def _api_key() -> str | None:
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    env = Path(__file__).resolve().parents[2] / ".." / "quant-airflow" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--stratify", default="month,sector,session")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="live judge's Haiku 4.5 snapshot")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--errors", type=Path, default=ERRORS)
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--chars-per-token", type=float, default=1.0)
    ap.add_argument("--est-output-tokens", type=int, default=200)
    ap.add_argument("--calibrate", type=int, default=0, help="count_tokens on N sampled requests")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--adopt-batch",
        default=None,
        help="record an existing batch for a failed (uncertain) create, then resume",
    )
    ap.add_argument(
        "--new-run",
        action="store_true",
        help="draw a fresh sample even though the state file holds a completed run",
    )
    args = ap.parse_args()
    if not args.model.startswith("claude-haiku-4-5"):
        sys.exit(f"model must be Haiku 4.5 (plan Global Constraints), got {args.model}")

    pairs = [json.loads(line) for line in args.pairs.open() if line.strip()]
    with httpx.Client() as client:
        judge = BatchJudge(
            client=client,
            api_key=None if args.dry_run else _api_key(),
            model=args.model,
            cache=args.cache,
            errors=args.errors,
            state=args.state,
            budget_usd=args.budget_usd,
            chars_per_token=args.chars_per_token,
            est_output_tokens=args.est_output_tokens,
            chunk=args.chunk,
        )
        try:
            if args.adopt_batch:
                judge.adopt_batch(args.adopt_batch)
            summary = judge.run(
                pairs,
                sample=args.sample,
                stratify=[f for f in args.stratify.split(",") if f],
                seed=args.seed,
                dry_run=args.dry_run,
                calibrate=args.calibrate,
                new_run=args.new_run,
            )
        except (BudgetExceeded, SubmitUncertain) as exc:
            sys.exit(f"ABORT: {exc}")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
