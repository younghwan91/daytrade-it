"""Time-ordered monthly-fold evaluation of entry-selection rules on per-pair rewards.

Fold m trains on every row strictly BEFORE month m and evaluates on month m, so no
fold shares a training row with its evaluation set (unlike LODO, where 40 folds
share 39/40 of their training days and are one model forty times over).

Two readouts per fold:

1. **Gate** -- mean reward of the pairs the rule would enter, against the mean over
   every pair in the live candidate pool that month. This is the sampling-robust
   number: with a ~30% stratified sample of each day's pairs, the *conditional*
   mean reward given decision-time features is still unbiased, so a selection
   rule can be scored even though no day is fully judged.

2. **Top-K/day** -- the fixed-capital comparison. Live enters ~5 positions a day and
   its cash is gone long before the candidate list is, so it effectively takes the
   first candidates of the day. Baselines: `first` (chronological, closest to live)
   and `pool` (the day's mean, i.e. a random pick). Candidate: the K highest-scoring
   pairs under the rule. Day P&L uses a flat 100,000 KRW notional per position --
   the A7-prep baseline's -465,246 KRW over 456 trades at -0.94% mean return implies
   a ~108,500 KRW average buy value, so this is calibrated, not arbitrary.

What neither readout captures: capital contention and the concurrent-position cap
across a *whole* day's candidates. Those need fully judged days.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from daytrade_it.application.trading import bandit

NOTIONAL = 100_000.0
TOP_K = 5
LAMS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = [r for r in rows if r["reward"] is not None and r["features"] is not None]
    rows.sort(key=lambda r: (r["day"], r["decide_at"]))
    return rows


def fit_rows(rows: list[dict], names: tuple[str, ...], lam: float) -> bandit.BanditModel:
    X = np.array([[r["features"][f] for f in names] for r in rows], dtype=float)
    y = np.array([r["reward"] for r in rows], dtype=float)
    return bandit.fit(X, y, lam=lam, feature_names=names)


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------- rule families


class Rule:
    """A trained selection rule: `score(row)` ranks, `enter(row)` gates."""

    name = "?"

    def score(self, row: dict) -> float:
        raise NotImplementedError

    def enter(self, row: dict) -> bool:
        raise NotImplementedError

    def describe(self) -> dict:
        return {}


class BanditRule(Rule):
    def __init__(self, names: tuple[str, ...], train: list[dict], label: str) -> None:
        self.name = label
        self.names = names
        # Inner split, also time-ordered: the last quarter of the training days
        # validates, the rest fits. (A month boundary would leave fold 2026-04,
        # whose training set is one month, with nothing to fit on.)
        days = sorted({r["day"] for r in train})
        cut = days[max(1, int(len(days) * 0.75))]
        inner_fit = [r for r in train if r["day"] < cut]
        inner_val = [r for r in train if r["day"] >= cut]
        best = None
        for lam in LAMS:
            m = fit_rows(inner_fit, names, lam)
            sel = [r["reward"] for r in inner_val if bandit.decide(m, r["features"])[0]]
            # Objective: total reward of the entered pairs on the held-out month.
            obj = float(np.sum(sel)) if sel else 0.0
            if best is None or obj > best[0]:
                best = (obj, lam)
        self.lam = best[1]
        self.model = fit_rows(train, names, self.lam)

    def score(self, row: dict) -> float:
        return self.model.predict({f: row["features"][f] for f in self.names})[0]

    def enter(self, row: dict) -> bool:
        return bandit.decide(self.model, {f: row["features"][f] for f in self.names})[0]

    def describe(self) -> dict:
        w = self.model.w
        return {
            "lam": self.lam,
            "intercept": float(w[0]),
            "coef": {n: float(c) for n, c in zip(self.names, w[1:], strict=True)},
            "signs": {n: ("+" if c > 0 else "-") for n, c in zip(self.names, w[1:], strict=True)},
        }


class MonotoneRule(Rule):
    """Sum of standardized features, each signed by its own train-set correlation.

    The minimal-variance alternative to a fitted linear model: no matrix inverse,
    no coefficient magnitudes, only a direction per feature learned on the train set.
    """

    def __init__(self, names: tuple[str, ...], train: list[dict], label: str) -> None:
        self.name = label
        self.names = names
        X = np.array([[r["features"][f] for f in names] for r in train], dtype=float)
        y = np.array([r["reward"] for r in train], dtype=float)
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd < 1e-9, np.inf, sd)
        Z = (X - self.mu) / self.sd
        self.sign = np.array(
            [1.0 if np.corrcoef(Z[:, i], y)[0, 1] > 0 else -1.0 for i in range(len(names))]
        )
        scores = self._scores(Z)
        # Gate at the train median: enter the better-scoring half.
        self.cut = float(np.median(scores))

    def _scores(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.sign

    def score(self, row: dict) -> float:
        x = np.array([row["features"][f] for f in self.names])
        return float(((x - self.mu) / self.sd) @ self.sign)

    def enter(self, row: dict) -> bool:
        return self.score(row) >= self.cut

    def describe(self) -> dict:
        return {
            "signs": {
                n: ("+" if s > 0 else "-") for n, s in zip(self.names, self.sign, strict=True)
            },
            "cut": self.cut,
        }


class ThresholdRule(Rule):
    """One feature, one threshold, direction and cut both fixed by hand (not fitted)."""

    def __init__(self, feature: str, cut: float, above: bool, label: str) -> None:
        self.name = label
        self.feature = feature
        self.cut = cut
        self.above = above

    def score(self, row: dict) -> float:
        v = row["features"][self.feature]
        return v if self.above else -v

    def enter(self, row: dict) -> bool:
        v = row["features"][self.feature]
        return v >= self.cut if self.above else v <= self.cut

    def describe(self) -> dict:
        return {"feature": self.feature, "cut": self.cut, "above": self.above}


FAMILIES: dict[str, tuple[str, ...]] = {
    "bandit13": bandit.FEATURES,
    "bandit5": ("score", "surprise", "pre_entry_move", "minutes_since_open", "ret_5m"),
    "bandit3": ("minutes_since_open", "ret_5m", "pre_entry_move"),
    "bandit2": ("minutes_since_open", "ret_5m"),
}


def build_rules(train: list[dict]) -> list[Rule]:
    rules: list[Rule] = []
    for label, names in FAMILIES.items():
        rules.append(BanditRule(names, train, label))
    rules.append(MonotoneRule(FAMILIES["bandit3"], train, "mono3"))
    rules.append(MonotoneRule(FAMILIES["bandit2"], train, "mono2"))
    rules.append(ThresholdRule("premarket", 0.0, above=False, label="intraday_only"))
    rules.append(ThresholdRule("minutes_since_open", 60.0, above=True, label="after_10am"))
    rules.append(ThresholdRule("minutes_since_open", 120.0, above=True, label="after_11am"))
    return rules


# ---------------------------------------------------------------- evaluation


def day_records(rows: list[dict], pick) -> list[dict]:  # noqa: ANN001
    """One day record per day, over the K pairs `pick` returns for that day."""
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    out = []
    for day in sorted(by_day):
        chosen = pick(by_day[day])
        rewards = [c["reward"] for c in chosen]
        out.append(
            {
                "day": day,
                "net": NOTIONAL * float(sum(rewards)),
                "trades": len(chosen),
                "losers": sum(1 for v in rewards if v < 0),
                "worst_trade": NOTIONAL * min(rewards) if rewards else 0.0,
                "errors": 0,
                "leftovers": 0,
            }
        )
    return out


def evaluate_fold(rule: Rule, ev: list[dict]) -> dict:
    sel = [r for r in ev if rule.enter(r)]
    pool_mean = mean([r["reward"] for r in ev])
    sel_mean = mean([r["reward"] for r in sel])
    cand = day_records(ev, lambda rs: sorted(rs, key=rule.score, reverse=True)[:TOP_K])
    return {
        "n_eval": len(ev),
        "n_selected": len(sel),
        "gate_mean_bp": sel_mean * 1e4,
        "pool_mean_bp": pool_mean * 1e4,
        "gate_delta_bp": (sel_mean - pool_mean) * 1e4,
        "topk_net": sum(d["net"] for d in cand),
        "days": cand,
    }


def _reward(row: dict) -> float:
    return float(row["reward"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from bayes_judge import judge

    rows = load(args.dataset)
    months = sorted({r["day"][:7] for r in rows})
    folds = months[1:]

    # Fixed baselines, computed once over the evaluated months.
    ev_all = [r for r in rows if r["day"][:7] in folds]
    base_first = day_records(ev_all, lambda rs: rs[:TOP_K])
    # No-skill comparator at the SAME trade count: the expected result of picking K
    # of the day's pairs at random, i.e. K x the day's mean. (Comparing against the
    # whole pool would let any rule "win" merely by trading ~9x less.)
    base_rand = []
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in ev_all:
        by_day[r["day"]].append(r)
    for day in sorted(by_day):
        rs = [x["reward"] for x in by_day[day]]
        k = min(TOP_K, len(rs))
        base_rand.append(
            {
                "day": day,
                "net": NOTIONAL * k * float(np.mean(rs)),
                "trades": k,
                "losers": int(round(k * float(np.mean([v < 0 for v in rs])))),
                "worst_trade": NOTIONAL * min(rs),
                "errors": 0,
                "leftovers": 0,
            }
        )

    print(f"rows={len(rows)}  months={months}  folds={folds}")
    print(
        f"pool mean over evaluated months = {mean([r['reward'] for r in ev_all]) * 1e4:+.0f} bp"
        f"  ({len(ev_all)} pairs)"
    )
    bf = sum(d["net"] for d in base_first)
    print(
        f"baseline 'first {TOP_K}/day': net {bf:,.0f} KRW over {len(base_first)} days"
        f"  ({sum(d['trades'] for d in base_first)} trades,"
        f" {sum(d['losers'] for d in base_first) / max(1, sum(d['trades'] for d in base_first)):.3f} losers)"
    )

    results: dict = {"months": months, "folds": {}, "rules": defaultdict(dict)}
    all_days: dict[str, list[dict]] = defaultdict(list)

    for m in folds:
        train = [r for r in rows if r["day"][:7] < m]
        ev = [r for r in rows if r["day"][:7] == m]
        rules = build_rules(train)
        print(f"\n=== fold {m}: train {len(train)} pairs (< {m}), eval {len(ev)} pairs")
        print(
            f"{'rule':16s} {'lam':>6s} {'sel':>6s} {'gate bp':>9s} {'pool bp':>9s}"
            f" {'delta':>8s} {'top5 net':>11s}"
        )
        for rule in rules:
            res = evaluate_fold(rule, ev)
            desc = rule.describe()
            all_days[rule.name].extend(res["days"])
            results["rules"][rule.name][m] = {
                "n_eval": res["n_eval"],
                "n_selected": res["n_selected"],
                "gate_mean_bp": res["gate_mean_bp"],
                "pool_mean_bp": res["pool_mean_bp"],
                "gate_delta_bp": res["gate_delta_bp"],
                "topk_net": res["topk_net"],
                "describe": desc,
            }
            lam = f"{desc['lam']:.0f}" if "lam" in desc else "-"
            print(
                f"{rule.name:16s} {lam:>6s} {res['n_selected']:>6d}"
                f" {res['gate_mean_bp']:>+9.0f} {res['pool_mean_bp']:>+9.0f}"
                f" {res['gate_delta_bp']:>+8.0f} {res['topk_net']:>+11,.0f}"
            )

    # Pooled out-of-sample totals per rule.
    print("\n=== pooled out-of-sample (all 5 folds)")
    print(f"{'rule':16s} {'sel':>6s} {'gate bp':>9s} {'top5 net':>12s} {'bp/trade':>9s}")
    for name, days in all_days.items():
        sel_n = sum(results["rules"][name][m]["n_selected"] for m in folds)
        sel_sum = sum(
            results["rules"][name][m]["n_selected"] * results["rules"][name][m]["gate_mean_bp"]
            for m in folds
            if results["rules"][name][m]["n_selected"]
        )
        gate = sel_sum / sel_n if sel_n else float("nan")
        net = sum(d["net"] for d in days)
        trades = sum(d["trades"] for d in days)
        print(
            f"{name:16s} {sel_n:>6d} {gate:>+9.0f} {net:>+12,.0f}"
            f" {net / trades / NOTIONAL * 1e4:>+9.0f}"
        )
    bfn = sum(d["net"] for d in base_first)
    bft = sum(d["trades"] for d in base_first)
    brn = sum(d["net"] for d in base_rand)
    brt = sum(d["trades"] for d in base_rand)
    print(
        f"{'BASE first5':16s} {'-':>6s} {'-':>9s} {bfn:>+12,.0f} {bfn / bft / NOTIONAL * 1e4:>+9.0f}"
    )
    print(
        f"{'BASE rand5':16s} {'-':>6s} {'-':>9s} {brn:>+12,.0f} {brn / brt / NOTIONAL * 1e4:>+9.0f}"
    )

    # In-sample ceiling: fit and evaluate a rule family on the SAME month. Not a
    # result -- an upper bound on what these 13 features could ever deliver, to
    # separate "the model is too weak" from "the features carry nothing".
    print("\n=== in-sample ceiling (fit and scored on the same month -- NOT out-of-sample)")
    print(f"{'month':9s} {'pool bp':>9s} {'best-5/day bp':>14s} {'oracle top-5 bp':>16s}")
    for m in months:
        ev = [r for r in rows if r["day"][:7] == m]
        rule = BanditRule(bandit.FEATURES, ev, "insample")
        ins = day_records(ev, lambda rs, _r=rule: sorted(rs, key=_r.score, reverse=True)[:TOP_K])
        oracle = day_records(ev, lambda rs: sorted(rs, key=_reward, reverse=True)[:TOP_K])
        pm = mean([r["reward"] for r in ev]) * 1e4
        ib = sum(d["net"] for d in ins) / sum(d["trades"] for d in ins) / NOTIONAL * 1e4
        ob = sum(d["net"] for d in oracle) / sum(d["trades"] for d in oracle) / NOTIONAL * 1e4
        print(f"{m:9s} {pm:>+9.0f} {ib:>+14.0f} {ob:>+16.0f}")

    # bayes_judge, top-K per day, against both baselines.
    print("\n=== bayes_judge (top-5/day, out-of-sample days pooled across folds)")
    verdicts = {}
    for name, days in all_days.items():
        cand = {d["day"]: d for d in days}
        for bname, base in (("first5", base_first), ("rand5", base_rand)):
            bmap = {d["day"]: d for d in base if d["day"] in cand}
            if set(bmap) != set(cand):
                continue
            v = judge(cand, bmap)
            verdicts[f"{name}_vs_{bname}"] = v
            print(
                f"{name:16s} vs {bname:6s}  p(mu>0)={v['p_mu_gt_0']:.3f}"
                f" mu={v['mu_mean']:+9.0f} days>=0={v['days_ge_0']}/{v['n_days']}"
                f" worst {v['worst_day_candidate']:+,.0f} vs {v['worst_day_baseline']:+,.0f}"
                f" losers {v['loser_ratio_candidate']:.3f} vs {v['loser_ratio_baseline']:.3f}"
                f"  adopt={v['adopt']}"
            )

    # Sign stability of the fitted linear models across folds.
    print("\n=== coefficient signs across time-ordered folds")
    for name in FAMILIES:
        per_fold = [results["rules"][name][m]["describe"]["signs"] for m in folds]
        feats = list(per_fold[0])
        print(f"-- {name}")
        for f in feats:
            seq = "".join(d[f] for d in per_fold)
            stab = max(seq.count("+"), seq.count("-")) / len(seq)
            print(f"   {f:26s} {seq}  {stab:.2f}")
        lams = [results["rules"][name][m]["describe"]["lam"] for m in folds]
        print(f"   lambda per fold: {lams}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"folds": results["rules"], "verdicts": verdicts}, indent=1, default=str)
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
