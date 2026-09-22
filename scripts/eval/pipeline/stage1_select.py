"""Stage 1: learn which news carries large upside -- from the TEXT -- as out-of-fold scores.

    # full run: search, shuffle-noise control, confirm read, final model, OOF file
    .venv/bin/python scripts/eval/pipeline/stage1_select.py \
        --out data/eval/pipeline/stage1_oof.jsonl --model-dir data/eval/pipeline/stage1_model
    # stage 4: score given days with the saved final model, decision-time inputs only
    .venv/bin/python scripts/eval/pipeline/stage1_select.py --score-only \
        --days 2026-08-03,2026-08-04 --out scores.jsonl --model-dir data/eval/pipeline/stage1_model

Inputs (build them first):
  scripts/eval/pipeline/stage1_build.py  -> data/eval/pipeline/stage1/{pairs,texts}.jsonl
  scripts/eval/pipeline/stage1_embed.py  -> embedding cache (runs in ~/venvs/emb, see its docstring)

PRE-REGISTERED DESIGN (fixed before any model was fitted; do not edit after results)
-------------------------------------------------------------------------------
Targets (per pair; up = max(mfe,0), down = max(-mae,0) -- mfe can be < 0 when the
entry quote is the last completed close and price never trades back to it):
  mfe, ret_close                 raw upside
  mfe_n, ret_n                   the same / (own_vol * sqrt(h / 390)): own_vol = 60-day daily
                                 log-return std before the day, h = minutes from the acting
                                 minute to the 15:30 close. The sqrt(h) horizon term was added
                                 after an 8%-sample DEBUG run of the code path (before any
                                 full run): with own_vol alone the numeric arm scored IC 0.36 on
                                 mfe/own_vol, almost all from act_min -- a 15:00 pair has 30
                                 minutes to make an MFE, a 09:00 pair 390. That is the clock,
                                 not the news; this normalises the expected move over the
                                 remaining holding window, which is what "own volatility" means
                                 for a same-day exit.
  rr   = up / (up + down)        in [0,1]; both zero -> 0.5
  lrr  = log((up + e) / (down + e)),  e = 0.001 (10 bp)
Pairs whose own_vol is undefined (< 20 prior returns) are excluded from fitting
and evaluation for EVERY target (so raw and normalised use one sample) but
still receive scores.
Fitting target: the within-day centred rank of the target (in (-0.5, 0.5)).
Aligned with the metric, robust to the heavy right tail of mfe.

Arms (feature blocks; ridge = closed form, intercept, multi-output):
  num        6 decision-time features, quantile-binned one-hot + missing flags
  tfidf      char_wb 2-4-gram TF-IDF (sublinear, min_df 5, 20k features, fit per window)
  emb        frozen multilingual-e5-small embedding (z-scored) + ridge
  emb_mlp    same embedding -> sklearn MLPRegressor(128 relu) (the deep head)
  emb_num    emb + num, ridge (num block nearly unpenalised)
  tfidf_num  tfidf + num, ridge (num block nearly unpenalised)
Hyperparameters (chosen per fold on an inner split, see below):
  num lam {1e-3,1e-1,10}; emb/emb_num lam {0.1,1,10} (alpha = lam*n);
  tfidf/tfidf_num lam {1,10,100} (alpha = lam*mean Gram diagonal);
  emb_mlp L2 alpha {1e-2, 1}; 15 epochs, batch 512, adam 1e-3, no early stopping.
  => 17 HP configs x 6 targets = 102 inner configs per window.

Validation:
  * months of stage1.train_range, expanding window. Month j is scored by the
    model fitted on months < j. First 3 months are burn-in (no score).
  * nested: the HP used to score month j is the one whose model fitted on
    months < j-1 had the best mean within-day IC on month j-1.
  * primary metric: mean within-day Spearman IC (days with >= 10 pairs),
    SE by bootstrap over days.
  * eligible candidates for the ONE chosen configuration: 6 arms x 4
    scale-free targets {mfe_n, ret_n, rr, lrr} = 24; chosen by selection-range
    OOF IC only. Raw targets are reported, never chosen.
  * confirm_range is read once for the chosen configuration.
  * shuffle control: targets permuted within day (all six jointly), the
    identical search (every arm, every HP, the nested choice, best-of-24),
    20 draws.
"""
# ruff: noqa: B023  -- `record` is a closure called inside the same window iteration that binds pe/ev/k; never stored.

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pickle
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/eval"))

import config  # noqa: E402

DATA = REPO / "data/eval/pipeline/stage1"
TARGETS = ["mfe", "ret_close", "mfe_n", "ret_n", "rr", "lrr"]
PRIMARY_TARGETS = ["mfe_n", "ret_n", "rr", "lrr"]
LRR_EPS = 0.001
ARMS = ["num", "tfidf", "emb", "emb_mlp", "emb_num", "tfidf_num"]
HP = {
    "num": [1e-3, 1e-1, 10.0],
    "tfidf": [1.0, 10.0, 100.0],
    "emb": [0.1, 1.0, 10.0],
    "emb_mlp": [1e-2, 1.0],
    "emb_num": [0.1, 1.0, 10.0],
    "tfidf_num": [1.0, 10.0, 100.0],
}
NUM_PEN = 1e-3  # alpha = NUM_PEN * n on the num block inside the text+num arms
CONT = ["act_min", "log_vol", "log_prev_tv", "pre_move", "ret_5m"]
TFIDF_KW = {
    "analyzer": "char_wb",
    "ngram_range": (2, 4),
    "min_df": 5,
    "max_features": 20000,
    "sublinear_tf": True,
    "dtype": np.float32,
}
MIN_DAY_N = 10
SESSION_MIN = 390  # 09:00 -> 15:30


# ============================================================================ data
def load_pairs(path: Path, days: set[str] | None = None, strip_labels: bool = False) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if strip_labels:
                r.pop("label", None)
            if days is not None and r["day"] not in days:
                continue
            rows.append(r)
    return rows


def target_matrix(rows: list[dict]) -> np.ndarray:
    T = np.full((len(rows), len(TARGETS)), np.nan)
    for i, r in enumerate(rows):
        lab, vol = r["label"], r["features"]["own_vol"]
        mfe, mae, rc = lab["mfe"], lab["mae"], lab["ret_close"]
        T[i, 0], T[i, 1] = mfe, rc
        if vol:
            scale = vol * math.sqrt(horizon_min(r["features"]["act_min"]) / SESSION_MIN)
            T[i, 2], T[i, 3] = mfe / scale, rc / scale
        # mfe < 0 (or mae > 0) happens when the entry is the last completed close
        # and the stock never trades back through it: clip each leg at zero.
        up, down = max(mfe, 0.0), max(-mae, 0.0)
        T[i, 4] = 0.5 if up + down == 0 else up / (up + down)
        T[i, 5] = math.log((up + LRR_EPS) / (down + LRR_EPS))
    return T


def horizon_min(act_min: int) -> int:
    """Minutes from the acting minute to the 15:30 close -- known at decision time."""
    return max(SESSION_MIN - int(act_min), 1)


def centred_rank(a: np.ndarray) -> np.ndarray:
    return (rankdata(a, axis=0) - 0.5) / a.shape[0] - 0.5


class NumEncoder:
    """Quantile-binned one-hot (5 bins, training-window edges) + missing flag per feature."""

    def fit(self, rows):
        self.edges = {}
        for f in CONT:
            v = np.array([r["features"][f] for r in rows if r["features"][f] is not None], float)
            self.edges[f] = (
                np.unique(np.quantile(v, [0.2, 0.4, 0.6, 0.8])) if len(v) else np.array([])
            )
        return self

    def transform(self, rows) -> np.ndarray:
        cols = []
        for f in CONT:
            e = self.edges[f]
            v = np.array([np.nan if r["features"][f] is None else r["features"][f] for r in rows])
            b = np.searchsorted(e, np.nan_to_num(v), side="right")
            oh = np.zeros((len(rows), len(e) + 2))
            ok = ~np.isnan(v)
            oh[np.arange(len(rows))[ok], b[ok]] = 1.0
            oh[~ok, -1] = 1.0
            cols.append(oh)
        cols.append(np.array([[float(r["features"]["premarket"])] for r in rows]))
        return np.hstack(cols)


class Std:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def load_embeddings(rows: list[dict], hash_key: str = "text_hash") -> np.ndarray:
    from stage1_embed import MODEL, cache_dir, load_cache

    try:
        idx, E = load_cache()
    except FileNotFoundError:
        sys.exit(
            f"embedding cache missing at {cache_dir()} -- build it with:\n"
            f"  OMP_NUM_THREADS=6 ~/venvs/emb/bin/python scripts/eval/pipeline/stage1_embed.py\n"
            f"(env setup in that script's docstring; model {MODEL})"
        )
    miss = [r[hash_key] for r in rows if r[hash_key] not in idx]
    if miss:
        sys.exit(
            f"{len(miss)} texts not in the embedding cache {cache_dir()} -- run:\n"
            f"  OMP_NUM_THREADS=6 ~/venvs/emb/bin/python scripts/eval/pipeline/stage1_embed.py"
        )
    return E[[idx[r[hash_key]] for r in rows]]


def load_texts() -> dict[str, str]:
    out = {}
    with (DATA / "texts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["hash"]] = r["text"]
    return out


# ============================================================================ models
def ridge_dense(X, Y, alphas_list, pen_mask=None):
    """Closed-form ridge on centred X for every alpha; pen_mask scales alpha per column."""
    G = X.T @ X
    XtY = X.T @ Y
    out = []
    if pen_mask is None:
        e, V = np.linalg.eigh(G)
        VtXY = V.T @ XtY
        for a in alphas_list:
            out.append(V @ (VtXY / (e + a)[:, None]))
    else:
        for a in alphas_list:
            A = G + np.diag(a * pen_mask)
            out.append(sla.solve(A, XtY, assume_a="pos"))
    return out


class TfidfBlock:
    """Per-window TF-IDF + closed-form ridge over all response columns at once.

    Rows sharing a text contribute identically to X'X, so the Gram is built on
    unique texts weighted by their multiplicity (exactly equal to the row-level
    Gram) -- one char n-gram fan-out article costs one row, not twenty.
    """

    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    def fit_gram(self, hashes: list[str], N: np.ndarray | None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        uniq = sorted(set(hashes))
        pos = {h: i for i, h in enumerate(uniq)}
        self.vec = TfidfVectorizer(**TFIDF_KW).fit([self.texts[h] for h in uniq])
        if hasattr(self.vec, "stop_words_"):  # every pruned n-gram; not needed to transform
            self.vec.stop_words_ = None
        Xu = self.vec.transform([self.texts[h] for h in uniq]).tocsr()
        ui = np.array([pos[h] for h in hashes])
        n = len(hashes)
        self.M = sp.csr_matrix((np.ones(n, np.float32), (ui, np.arange(n))), shape=(len(uniq), n))
        c = np.asarray(self.M.sum(1)).ravel().astype(np.float32)
        Xw = sp.csr_matrix(Xu.multiply(c[:, None]))
        XuT = Xu.T.tocsr()
        p = Xu.shape[1]
        G = np.zeros((p, p), np.float32)
        for j0 in range(0, p, 4000):
            G[:, j0 : j0 + 4000] = (XuT @ Xw[:, j0 : j0 + 4000]).toarray()
        self.mu = (np.asarray(Xw.sum(0)).ravel() / n).astype(np.float32)
        G -= n * np.outer(self.mu, self.mu)
        self.G, self.Xu, self.n = G, Xu, n
        self.XtN = (
            None
            if N is None
            else (XuT @ (self.M @ N)).astype(np.float32) - n * np.outer(self.mu, N.mean(0))
        )
        self.mean_diag = float(np.mean(np.diag(G)))
        return self

    def solve(self, Y: np.ndarray, lam: float, N: np.ndarray | None):
        """Returns (w_text, w_num or None) for tfidf-only and, if N given, tfidf+num."""
        XtY = (self.Xu.T @ (self.M @ Y)).astype(np.float32)
        A = self.G.copy()
        A[np.diag_indices_from(A)] += lam * self.mean_diag
        cf = sla.cho_factor(A, overwrite_a=True, check_finite=False)
        w_t = sla.cho_solve(cf, XtY, check_finite=False)
        res = {"tfidf": (w_t, None)}
        if N is not None:
            B = self.XtN
            AiB = sla.cho_solve(cf, B, check_finite=False)
            D = N.T @ N + NUM_PEN * self.n * np.eye(N.shape[1])
            S = D - B.T @ AiB
            NtY = N.T @ Y
            w_n = np.linalg.solve(S, NtY - B.T @ w_t)
            w_tn = w_t - AiB @ w_n
            res["tfidf_num"] = (w_tn, w_n)
        del A, cf
        return res

    def predict(self, hashes, w_t, N=None, w_n=None):
        X = self.vec.transform([self.texts[h] for h in hashes])
        P = X @ w_t - self.mu @ w_t
        if w_n is not None:
            P = P + N @ w_n
        return np.asarray(P, dtype=np.float32)


def fit_mlp(X, Y, alpha, seed):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor

    m = MLPRegressor(
        hidden_layer_sizes=(128,),
        alpha=alpha,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=15,
        early_stopping=False,
        random_state=seed,
        shuffle=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(X, Y)
    return m


# ============================================================================ metric
def daily_ic(P: np.ndarray, Tm: np.ndarray, day_of: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Within-day Spearman IC per column. P, Tm: (n, C) aligned columns."""
    days, out = [], []
    order = np.argsort(day_of, kind="stable")
    ds = day_of[order]
    cuts = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:], strict=False):
        if b - a < MIN_DAY_N:
            continue
        idx = order[a:b]
        pr = rankdata(P[idx], axis=0)
        tr = rankdata(Tm[idx], axis=0)
        pr -= pr.mean(0)
        tr -= tr.mean(0)
        den = np.sqrt((pr**2).sum(0) * (tr**2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, (pr * tr).sum(0) / den, 0.0))
        days.append(ds[a])
    return days, np.array(out)


def boot_se(x: np.ndarray, n=2000, seed=0) -> float:
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(np.std(x[rng.integers(0, len(x), (n, len(x)))].mean(1)))


# ============================================================================ search
def run_search(
    rows,
    T,
    month_of,
    months,
    n_draws,
    seed,
    model_dir: Path,  # noqa: ARG001 -- reserved for per-window model dumps
    log,
    arms=ARMS,
    hash_key="text_hash",
    real_only=False,
):
    """Window loop. Returns per (arm,hp) per window: daily IC arrays for month k, and
    real-draw predictions. Saves the final (all-months) window models."""
    n = len(rows)
    lab = np.array([r["label"] is not None for r in rows])
    ok = lab & ~np.isnan(T[:, 2])
    day_of = np.array([r["day"] for r in rows])
    draws = 1 if real_only else n_draws + 1

    # within-day permutations of the evaluable rows (draw 0 = identity)
    rng = np.random.default_rng(seed)
    perms = [np.arange(n)]
    ok_idx_by_day = defaultdict(list)
    for i in np.flatnonzero(ok):
        ok_idx_by_day[day_of[i]].append(i)
    for _ in range(1, draws):
        p = np.arange(n)
        for idx in ok_idx_by_day.values():
            idx = np.array(idx)
            p[idx] = idx[rng.permutation(len(idx))]
        perms.append(p)
    C = draws * len(TARGETS)
    Tall = np.full((n, C), np.nan, np.float32)  # realised (permuted) targets, column d*6+t
    Rall = np.zeros((n, C), np.float32)  # within-day centred ranks = fitting targets
    for d, p in enumerate(perms):
        Td = T[p]
        Td[~ok] = np.nan
        Tall[:, d * 6 : (d + 1) * 6] = Td
        for idx in ok_idx_by_day.values():
            Rall[idx, d * 6 : (d + 1) * 6] = centred_rank(Td[idx])

    need_emb = any(a.startswith("emb") for a in arms)
    need_tf = any(a.startswith("tfidf") for a in arms)
    E = load_embeddings(rows, hash_key) if need_emb else None
    texts = load_texts() if need_tf else None
    hashes = [r[hash_key] for r in rows]

    K = len(months)
    ic = {}  # (arm, hp_i, k) -> (days, array[n_days, C])
    preds = {}  # (arm, hp_i) -> array (n, 6) real draw
    for arm in arms:
        for h in range(len(HP[arm])):
            preds[(arm, h)] = np.full((n, len(TARGETS)), np.nan, np.float32)
    final = {}
    hold = np.array([r["label"] is None for r in rows])
    for k in range(2, K + 1):
        t0 = time.time()
        tr = np.flatnonzero(ok & (month_of < k))
        pe = np.flatnonzero(month_of == k) if k < K else np.flatnonzero(hold)
        if len(pe) == 0:
            continue
        ev = pe[ok[pe]]  # evaluable subset of the predicted rows
        Ytr = Rall[tr] if k < K else Rall[tr][:, :6]
        tr_rows = [rows[i] for i in tr]
        pe_rows = [rows[i] for i in pe]
        enc = NumEncoder().fit(tr_rows)
        nstd = Std().fit(enc.transform(tr_rows))
        Ntr = nstd.transform(enc.transform(tr_rows))
        Npe = nstd.transform(enc.transform(pe_rows))

        def record(arm, h, P):
            preds[(arm, h)][pe] = P[:, :6]
            if k < K:
                pos = {i: j for j, i in enumerate(pe)}
                sel = np.array([pos[i] for i in ev])
                days, arr = daily_ic(P[sel], Tall[ev][:, : P.shape[1]], day_of[ev])
                ic[(arm, h, k)] = (days, arr)

        if "num" in arms:
            Ws = ridge_dense(Ntr, Ytr, [lam * len(tr) for lam in HP["num"]])
            for h, W in enumerate(Ws):
                record("num", h, Npe @ W)
                if k == K:
                    final[("num", h)] = {"enc": enc, "nstd": nstd, "W": W[:, :6]}
        if need_emb:
            est = Std().fit(E[tr])
            Etr, Epe = est.transform(E[tr]), est.transform(E[pe])
            if "emb" in arms:
                Ws = ridge_dense(Etr, Ytr, [lam * len(tr) for lam in HP["emb"]])
                for h, W in enumerate(Ws):
                    record("emb", h, Epe @ W)
                    if k == K:
                        final[("emb", h)] = {"est": est, "W": W[:, :6]}
            if "emb_num" in arms:
                X = np.hstack([Etr, Ntr])
                mask = np.r_[np.ones(Etr.shape[1]), np.zeros(Ntr.shape[1])]
                Ws = []
                for lam in HP["emb_num"]:
                    pen = lam * len(tr) * mask + NUM_PEN * len(tr) * (1 - mask)
                    Ws.append(ridge_dense(X, Ytr, [1.0], pen_mask=pen)[0])
                for h, W in enumerate(Ws):
                    record("emb_num", h, np.hstack([Epe, Npe]) @ W)
                    if k == K:
                        final[("emb_num", h)] = {
                            "est": est,
                            "enc": enc,
                            "nstd": nstd,
                            "W": W[:, :6],
                        }
            if "emb_mlp" in arms:
                for h, a in enumerate(HP["emb_mlp"]):
                    P = np.zeros((len(pe), Ytr.shape[1]), np.float32)
                    for d in range(Ytr.shape[1] // 6):
                        m = fit_mlp(Etr, Ytr[:, d * 6 : (d + 1) * 6], a, seed + d)
                        P[:, d * 6 : (d + 1) * 6] = m.predict(Epe)
                        if k == K and d == 0:
                            final[("emb_mlp", h)] = {"est": est, "mlp": m}
                    record("emb_mlp", h, P)
            del Etr, Epe
        if need_tf:
            tb = TfidfBlock(texts).fit_gram([hashes[i] for i in tr], Ntr)
            hpe = [hashes[i] for i in pe]
            for h, lam in enumerate(HP["tfidf"]):
                res = tb.solve(Ytr, lam, Ntr if "tfidf_num" in arms else None)
                if "tfidf" in arms:
                    w_t, _ = res["tfidf"]
                    record("tfidf", h, tb.predict(hpe, w_t))
                    if k == K:
                        final[("tfidf", h)] = {"vec": tb.vec, "mu": tb.mu, "w_t": w_t[:, :6]}
                if "tfidf_num" in arms:
                    w_t, w_n = res["tfidf_num"]
                    record("tfidf_num", h, tb.predict(hpe, w_t, Npe, w_n))
                    if k == K:
                        final[("tfidf_num", h)] = {
                            "vec": tb.vec,
                            "mu": tb.mu,
                            "w_t": w_t[:, :6],
                            "enc": enc,
                            "nstd": nstd,
                            "w_n": w_n[:, :6],
                        }
            del tb
        log(
            f"window <{months[k] if k < K else 'end'}: train {len(tr)} rows, predicted {len(pe)}, "
            f"{time.time() - t0:.0f}s"
        )
    return ic, preds, final, Tall, ok, day_of


def assemble(ic, arms, K, draws, month_names):
    """Nested choice -> per (arm, target, draw) OOF daily IC series; also the HP chosen per month."""
    series = {}  # (arm, t, d) -> list[(day, ic)]
    chosen = defaultdict(dict)  # (arm, t) -> {month: hp} for draw 0
    for arm in arms:
        H = len(HP[arm])
        for k in range(3, K):
            inner = np.array(
                [np.nanmean(ic[(arm, h, k - 1)][1], axis=0) for h in range(H)]
            )  # H x C
            best = inner.argmax(0)  # per column
            days = ic[(arm, 0, k)][0]
            cube = np.stack([ic[(arm, h, k)][1] for h in range(H)])  # H x days x C
            pick = cube[best, :, np.arange(cube.shape[2])].T  # days x C
            for d in range(draws):
                for t in range(len(TARGETS)):
                    c = d * 6 + t
                    series.setdefault((arm, t, d), []).extend(zip(days, pick[:, c], strict=False))
            for t in range(len(TARGETS)):
                chosen[(arm, TARGETS[t])][month_names[k]] = HP[arm][int(best[t])]
    return series, chosen


def hp_final(ic, arm, K):
    """HP per target for the final (all-months) model: best on the last month."""
    H = len(HP[arm])
    inner = np.array([np.nanmean(ic[(arm, h, K - 1)][1][:, :6], axis=0) for h in range(H)])
    return inner.argmax(0)


def summarise(series_ad, rng_days):
    x = np.array([v for d, v in series_ad if rng_days[0] <= d <= rng_days[1]])
    return {
        "ic": float(x.mean()) if len(x) else None,
        "se": boot_se(x),
        "days": int(len(x)),
        "t": float(x.mean() / boot_se(x)) if len(x) > 1 else None,
    }


# ============================================================================ scoring (final model)
def score_rows(rows, model_dir: Path) -> np.ndarray:
    """Final-model scores from decision-time inputs only: (n, 6) per target with its chosen arm."""
    meta = json.loads((model_dir / "meta.json").read_text())
    with (model_dir / "final_models.pkl").open("rb") as f:
        models = pickle.load(f)
    need = {meta["final_arm_hp"][t][0] for t in TARGETS}
    E = load_embeddings(rows) if any(a.startswith("emb") for a in need) else None
    texts = load_texts() if any(a.startswith("tfidf") for a in need) else None
    out = np.full((len(rows), len(TARGETS)), np.nan, np.float32)
    cache = {}
    for ti, t in enumerate(TARGETS):
        arm, h = meta["final_arm_hp"][t]
        if (arm, h) not in cache:
            m = models[f"{arm}|{h}"]
            if arm == "num":
                P = m["nstd"].transform(m["enc"].transform(rows)) @ m["W"]
            elif arm == "emb":
                P = m["est"].transform(E) @ m["W"]
            elif arm == "emb_num":
                N = m["nstd"].transform(m["enc"].transform(rows))
                P = np.hstack([m["est"].transform(E), N]) @ m["W"]
            elif arm == "emb_mlp":
                P = m["mlp"].predict(m["est"].transform(E))
            else:
                X = m["vec"].transform([texts[r["text_hash"]] for r in rows])
                P = X @ m["w_t"] - m["mu"] @ m["w_t"]
                if arm == "tfidf_num":
                    P = P + m["nstd"].transform(m["enc"].transform(rows)) @ m["w_n"]
            cache[(arm, h)] = np.asarray(P, np.float32)
        out[:, ti] = cache[(arm, h)][:, ti]
    return out


def score_only(args):
    model_dir = Path(args.model_dir)
    days = set(args.days.split(",")) if args.days else None
    if days is None:
        sys.exit("--score-only needs --days")
    rows = load_pairs(Path(args.pairs), days=days, strip_labels=True)  # labels never read
    meta = json.loads((model_dir / "meta.json").read_text())
    S = score_rows(rows, model_dir)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r, s in zip(rows, S, strict=False):
            rec = {
                "key": r["key"],
                "news_id": r["news_id"],
                "code": r["code"],
                "day": r["day"],
                "decide_at": r["decide_at"],
                "month": r["month"],
                "trained_through": meta["trained_through"],
            }
            rec.update({f"score_{t}": float(v) for t, v in zip(TARGETS, s, strict=False)})
            rec["score"] = rec[f"score_{meta['chosen']['target']}"]
            f.write(json.dumps(rec) + "\n")
    print(f"scored {len(rows)} pairs over {len(days)} days -> {args.out}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cfg = config.load()
    ap.add_argument("--out", default=str(REPO / cfg["outputs"]["stage1_oof"]))
    ap.add_argument("--model-dir", default=str(REPO / cfg["outputs"]["stage1_model"]))
    ap.add_argument("--pairs", default=str(DATA / "pairs.jsonl"))
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--burnin-months", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--days", default=None, help="comma-separated days for --score-only")
    ap.add_argument(
        "--ablation",
        choices=["mask"],
        default=None,
        help="re-run the text arms on name-masked text (real labels only); needs a prior full run",
    )
    args = ap.parse_args()
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(min(args.threads, 6))
    except ImportError:
        pass
    if args.score_only:
        return score_only(args)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    logf = (model_dir / ("ablation_mask.log" if args.ablation else "run.log")).open("w")

    def log(msg):
        line = f"{dt.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    s1 = cfg["stage1"]
    t0d, t1d = s1["train_range"]
    held = config.holdout_days()
    rows = load_pairs(Path(args.pairs))
    # guard: every row carrying an outcome must be outside the frozen holdout
    config.assert_no_holdout_labels({r["day"] for r in rows if r.get("label") is not None}, stage=1)
    rows = [
        r for r in rows if (t0d <= r["day"] <= t1d and r["label"] is not None) or r["day"] in held
    ]
    rows.sort(key=lambda r: (r["day"], r["decide_at"], r["key"]))
    months = sorted({r["month"] for r in rows if r["label"] is not None})
    K = len(months)
    mi = {m: i for i, m in enumerate(months)}
    month_of = np.array([mi.get(r["month"], K) if r["label"] is not None else K for r in rows])
    lab_rows = [r for r in rows if r["label"] is not None]
    T = np.full((len(rows), len(TARGETS)), np.nan)
    lab_mask = np.array([r["label"] is not None for r in rows])
    T[lab_mask] = target_matrix(lab_rows)
    log(
        f"rows {len(rows)} (labelled {lab_mask.sum()}, holdout {(~lab_mask).sum()}), months {months}"
    )

    if args.ablation:
        return run_ablation(args, rows, T, month_of, months, model_dir, log)

    ic, preds, final, Tall, ok, day_of = run_search(
        rows, T, month_of, months, args.draws, args.seed, model_dir, log
    )
    draws = args.draws + 1
    series, chosen_hp = assemble(ic, ARMS, K, draws, months)
    sel_rng = s1["selection_range"]
    con_rng = s1["confirm_range"]
    burn = months[: args.burnin_months]

    # ---- the table (real draw) and the shuffle distribution ---------------
    table = {}
    for arm in ARMS:
        for t, tn in enumerate(TARGETS):
            table[f"{arm}|{tn}"] = {
                "selection": summarise(series[(arm, t, 0)], sel_rng),
                "hp_by_month": chosen_hp[(arm, tn)],
            }
    elig = [(a, TARGETS.index(t)) for a in ARMS for t in PRIMARY_TARGETS]
    all36 = [(a, t) for a in ARMS for t in range(len(TARGETS))]

    def best_of(d, cands):
        vals = [(summarise(series[(a, t, d)], sel_rng)["ic"], a, t) for a, t in cands]
        return max(vals)

    real_best = best_of(0, elig)
    shuffle = [best_of(d, elig)[0] for d in range(1, draws)]
    shuffle36 = [best_of(d, all36)[0] for d in range(1, draws)]
    per_cand_null = {
        f"{a}|{TARGETS[t]}": [summarise(series[(a, t, d)], sel_rng)["ic"] for d in range(1, draws)]
        for a, t in all36
    }
    chosen_arm, chosen_t = real_best[1], TARGETS[real_best[2]]
    log(f"chosen on selection range: {chosen_arm} / {chosen_t}  IC {real_best[0]:.4f}")

    # ---- confirm read, once ------------------------------------------------
    confirm = summarise(series[(chosen_arm, TARGETS.index(chosen_t), 0)], con_rng)
    confirm_all = {
        k: summarise(series[(a, t, 0)], con_rng)
        for k, (a, t) in ((f"{a}|{TARGETS[t]}", (a, t)) for a, t in all36)
    }

    # ---- per-target arm (selection range only) ------------------------------
    arm_for = {}
    for t, tn in enumerate(TARGETS):
        arm_for[tn] = max(ARMS, key=lambda a: summarise(series[(a, t, 0)], sel_rng)["ic"])
    arm_for[chosen_t] = chosen_arm

    # ---- OOF scores (real draw) ---------------------------------------------
    hp_month = {}  # (arm, t, month_index) -> hp index used to score that month
    for arm in ARMS:
        H = len(HP[arm])
        for k in range(3, K):
            inner = np.array([np.nanmean(ic[(arm, h, k - 1)][1][:, :6], axis=0) for h in range(H)])
            for t in range(len(TARGETS)):
                hp_month[(arm, t, k)] = int(inner[:, t].argmax())
    final_arm_hp = {}
    for t, tn in enumerate(TARGETS):
        a = arm_for[tn]
        final_arm_hp[tn] = [a, int(hp_final(ic, a, K)[t])]
    models = {f"{a}|{h}": final[(a, h)] for a, h in {tuple(v) for v in final_arm_hp.values()}}
    last_train_day = max(r["day"] for r in lab_rows)
    meta = {
        "chosen": {"arm": chosen_arm, "target": chosen_t},
        "arm_for_target": arm_for,
        "final_arm_hp": final_arm_hp,
        "trained_through": last_train_day,
        "months": months,
        "targets": TARGETS,
        "hp_grid": HP,
        "tfidf": {k: str(v) for k, v in TFIDF_KW.items()},
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    with (model_dir / "final_models.pkl").open("wb") as f:
        pickle.dump(models, f)

    # holdout rows: scored through the saved-model path (same code as --score-only)
    hold_idx = np.flatnonzero(~lab_mask)
    S_hold = score_rows([{**rows[i], "label": None} for i in hold_idx], model_dir)
    last_day_of_month = {}
    for r in lab_rows:
        last_day_of_month[r["month"]] = max(last_day_of_month.get(r["month"], ""), r["day"])
    n_split = Counter()
    oof = np.full((len(rows), len(TARGETS)), np.nan, np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        hpos = {i: j for j, i in enumerate(hold_idx)}
        for i, r in enumerate(rows):
            k = month_of[i]
            rec = {
                "key": r["key"],
                "news_id": r["news_id"],
                "code": r["code"],
                "day": r["day"],
                "decide_at": r["decide_at"],
                "month": r["month"],
            }
            if not lab_mask[i]:
                split, tt = "holdout", last_train_day
                sc = S_hold[hpos[i]]
            elif r["month"] in burn:
                split, tt, sc = "burnin", None, None
            else:
                split = "selection" if sel_rng[0] <= r["day"] <= sel_rng[1] else "confirm"
                tt = last_day_of_month[months[k - 1]]
                sc = np.array(
                    [
                        preds[(arm_for[tn], hp_month[(arm_for[tn], t, k)])][i, t]
                        for t, tn in enumerate(TARGETS)
                    ]
                )
                oof[i] = sc
            rec["split"] = split
            rec["trained_through"] = tt
            for t, tn in enumerate(TARGETS):
                rec[f"score_{tn}"] = None if sc is None else float(sc[t])
            rec["score"] = rec[f"score_{chosen_t}"]
            n_split[split] += 1
            f.write(json.dumps(rec) + "\n")
    log(f"OOF written {out}: {dict(n_split)}")

    # ---- checks: in-sample guard, economics, templates, vol ------------------
    report = {
        "n_rows": len(rows),
        "split_counts": dict(n_split),
        "months": months,
        "burnin": burn,
        "pairs_per_month": dict(Counter(r["month"] for r in rows)),
        "evaluable_rows(own_vol defined)": int(ok.sum()),
        "N": {
            "arms": len(ARMS),
            "targets": len(TARGETS),
            "eligible_candidates": len(elig),
            "all_candidates": len(all36),
            "hp_configs_per_window": sum(len(v) for v in HP.values()) * 6,
            "shuffle_draws": args.draws,
        },
        "table_selection": table,
        "chosen": {
            "arm": chosen_arm,
            "target": chosen_t,
            "selection": table[f"{chosen_arm}|{chosen_t}"]["selection"],
        },
        "shuffle_best_of_24": dist(shuffle, real_best[0]),
        "shuffle_best_of_36": dist(shuffle36, best_of(0, all36)[0]),
        "per_candidate_null": {
            k: dist(v, table[k]["selection"]["ic"]) for k, v in per_cand_null.items()
        },
        "confirm_chosen_read_once": confirm,
        "confirm_all_for_reference_not_used_for_choice": confirm_all,
        "arm_for_target": arm_for,
        "final_arm_hp": final_arm_hp,
    }
    report["text_vs_num"] = text_vs_num(series, sel_rng)
    report["economic_curve_chosen"] = econ_curve(
        rows, oof[:, TARGETS.index(chosen_t)], lab_mask, ok, sel_rng, con_rng
    )
    report["templates"] = template_check(
        rows, oof[:, TARGETS.index(chosen_t)], T, TARGETS.index(chosen_t), ok, sel_rng
    )
    report["vol_confound"] = vol_check(rows, T, ok)
    (model_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log(f"report -> {model_dir / 'report.json'}")


def dist(null, real):
    a = np.array([x for x in null if x is not None])
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.9)),
        "max": float(a.max()),
        "real": real,
        "real_rank_pct": float((a < real).mean()),
    }


def text_vs_num(series, rng):
    """Paired day-level difference: text(+num) arm minus num-only, per target."""
    out = {}
    for t, tn in enumerate(TARGETS):
        base = dict(series[("num", t, 0)])
        for arm in ARMS:
            if arm == "num":
                continue
            s = dict(series[(arm, t, 0)])
            dd = np.array([s[d] - base[d] for d in s if rng[0] <= d <= rng[1] and d in base])
            out[f"{arm}-num|{tn}"] = {
                "diff": float(dd.mean()),
                "se": boot_se(dd),
                "days": int(len(dd)),
            }
    return out


def econ_curve(rows, score, lab_mask, ok, sel_rng, con_rng):
    from costs import round_trip

    fracs = [round(x, 3) for x in np.r_[np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.0001, 0.05)]]
    net = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if lab_mask[i]:
            cost = round_trip(
                r["day"], r["market"] or "거래소", at_open=r["features"]["act_min"] < 5
            )
            net[i] = r["label"]["ret_close"] - cost
    res = {}
    for name, rng in (("selection", sel_rng), ("confirm", con_rng)):
        by_day = defaultdict(list)
        for i, r in enumerate(rows):
            if ok[i] and not np.isnan(score[i]) and rng[0] <= r["day"] <= rng[1]:
                by_day[r["day"]].append(i)
        curve = []
        for f in fracs:
            ex, top = [], []
            for idx in by_day.values():
                if len(idx) < MIN_DAY_N:
                    continue
                idx = np.array(idx)
                m = max(1, math.ceil(f * len(idx)))
                sel = idx[np.argsort(-score[idx], kind="stable")[:m]]
                top.append(net[sel].mean())
                ex.append(net[sel].mean() - net[idx].mean())
            ex, top = np.array(ex), np.array(top)
            curve.append(
                {
                    "frac": f,
                    "excess_bp": 1e4 * ex.mean(),
                    "excess_se_bp": 1e4 * boot_se(ex),
                    "net_top_bp": 1e4 * top.mean(),
                    "days": int(len(ex)),
                }
            )
        res[name] = curve
    return res


def template_check(rows, score, T, t, ok, rng):
    import re

    tag = re.compile(r"^\s*(\[[^\]]{1,20}\]|【[^】]{1,20}】)")
    texts = load_texts()
    tmpl = []
    for r in rows:
        title = texts[r["text_hash"]].split("\n", 1)[0]
        m = tag.match(title)
        tmpl.append(
            m.group(1)
            if m
            else ("<robo:뉴스핌 로보뉴스>" if "로보뉴스" in texts[r["text_hash"]] else "")
        )
    sel = np.array(
        [
            ok[i] and not np.isnan(score[i]) and rng[0] <= rows[i]["day"] <= rng[1]
            for i in range(len(rows))
        ]
    )
    fan = Counter(r["text_hash"] for r in rows)
    fan_by_article = Counter(r["news_id"] for r in rows)
    day_of = np.array([r["day"] for r in rows])

    def ic_of(mask):
        d, arr = daily_ic(score[mask][:, None], T[mask][:, t : t + 1], day_of[mask])
        x = arr[:, 0] if len(arr) else np.array([])
        return {
            "ic": float(x.mean()) if len(x) else None,
            "se": boot_se(x),
            "days": int(len(x)),
            "n": int(mask.sum()),
        }

    cnt = Counter(tmpl[i] for i in np.flatnonzero(sel))
    top = [k for k, _ in cnt.most_common(12) if k]
    tm = np.array(tmpl)
    out = {
        "pairs_eval": int(sel.sum()),
        "share_with_template_tag": float(np.mean([bool(tmpl[i]) for i in np.flatnonzero(sel)])),
        "top_templates": {k: cnt[k] for k in top},
        "text_fanout_hist": dict(Counter(min(v, 20) for v in fan.values())),
        "pairs_whose_text_is_shared_by_>1_code": float(
            np.mean([fan[r["text_hash"]] > 1 for r in rows])
        ),
        "articles": len(fan_by_article),
        "all": ic_of(sel),
        "single_code_texts_only": ic_of(sel & np.array([fan[r["text_hash"]] == 1 for r in rows])),
        "no_template": ic_of(sel & (tm == "")),
        "drop_one_template": {k: ic_of(sel & (tm != k)) for k in top},
        "within_template_only": {k: ic_of(sel & (tm == k)) for k in top[:5]},
    }
    return out


def vol_check(rows, T, ok):
    """Within-day Spearman of each target with own_vol: how much of it is just volatility."""
    vol = np.array([r["features"]["own_vol"] or np.nan for r in rows])
    act = np.array([float(r["features"]["act_min"]) for r in rows])
    day_of = np.array([r["day"] for r in rows])
    m = ok & ~np.isnan(vol)
    out = {}
    for name, x in (("own_vol", vol), ("act_min", act)):
        d, arr = daily_ic(np.repeat(x[m][:, None], len(TARGETS), 1), T[m], day_of[m])
        for i, tn in enumerate(TARGETS):
            out.setdefault(tn, {})[f"ic_with_{name}"] = float(arr[:, i].mean())
            out[tn][f"se_{name}"] = boot_se(arr[:, i])
    return out


def run_ablation(args, rows, T, month_of, months, model_dir, log):
    """Name-masking ablation, real labels only, identical nested protocol.

    Runs every arm twice -- on the plain text (reproduces draw 0 of the main
    run exactly, a determinism check) and on the name-masked text -- and saves
    the day-level OOF IC series of both, so text-over-numeric increments can be
    read on selection and on confirm. `num` does not read text; it is included
    as the baseline of the increments.
    """
    cfg = config.load()["stage1"]
    rep = json.loads((model_dir / "report.json").read_text())
    out = {"name_in_text_share": float(np.mean([r["name_in_text"] for r in rows])), "runs": {}}
    all_series = {}
    for key in ("text_hash", "masked_hash"):
        ic, *_ = run_search(
            rows,
            T,
            month_of,
            months,
            0,
            args.seed,
            model_dir,
            log,
            arms=ARMS,
            hash_key=key,
            real_only=True,
        )
        series, _ = assemble(ic, ARMS, len(months), 1, months)
        all_series[key] = {f"{a}|{TARGETS[t]}": v for (a, t, d), v in series.items()}
        res = {}
        for arm in ARMS:
            for t, tn in enumerate(TARGETS):
                res[f"{arm}|{tn}"] = {
                    "selection": summarise(series[(arm, t, 0)], cfg["selection_range"]),
                    "confirm": summarise(series[(arm, t, 0)], cfg["confirm_range"]),
                }
        inc = {}
        for rname in ("selection_range", "confirm_range"):
            rng = cfg[rname]
            for t, tn in enumerate(TARGETS):
                base = dict(series[("num", t, 0)])
                for arm in ARMS[1:]:
                    sd = dict(series[(arm, t, 0)])
                    dd = np.array(
                        [sd[d] - base[d] for d in sd if rng[0] <= d <= rng[1] and d in base]
                    )
                    inc[f"{rname}|{arm}-num|{tn}"] = {
                        "diff": float(dd.mean()),
                        "se": boot_se(dd),
                        "days": int(len(dd)),
                    }
        out["runs"][key] = {"table": res, "increment_over_num": inc}
        log(f"ablation run {key} done")
    repro = [
        abs(out["runs"]["text_hash"]["table"][k]["selection"]["ic"] - v["selection"]["ic"])
        for k, v in rep["table_selection"].items()
    ]
    out["max_abs_diff_vs_main_run_draw0"] = float(max(repro))
    (model_dir / "ablation_mask.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with (model_dir / "ablation_series.pkl").open("wb") as f:
        pickle.dump(all_series, f)
    log(f"ablation -> {model_dir / 'ablation_mask.json'}")


if __name__ == "__main__":
    main()
