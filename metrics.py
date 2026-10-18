"""Ranking, calibration and bootstrap helpers shared by every evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def topk(ranks, ks=config.TOP_K) -> dict:
    r = np.asarray(ranks, dtype=float)
    out = {f"top{k}": float((r <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    out["n"] = int(len(r))
    return out


def bootstrap_ci(values, stat=np.mean, n: int = 1000, seed: int = config.SEED, alpha: float = 0.05,
                 groups=None) -> tuple[float, float]:
    """Percentile interval; when groups are given, resample groups (cohorts) not cases."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if groups is None:
        stats = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    else:
        g = pd.Series(v).groupby(np.asarray(groups)).apply(list)
        keys = g.index.to_numpy()
        stats = []
        for _ in range(n):
            pick = rng.choice(keys, size=len(keys), replace=True)
            stats.append(stat(np.concatenate([g[k] for k in pick])))
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def calibration(prob, correct, bins: int = 10) -> dict:
    p = np.asarray(prob, dtype=float)
    y = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": conf, "accuracy": acc})
        ece += m.sum() / len(p) * abs(conf - acc)
    brier = float(((p - y) ** 2).mean())
    return {"bins": rows, "ece": float(ece), "brier": brier}


def table(results: dict[str, dict], keys=("top1", "top3", "top10", "top50", "mrr", "median_rank")) -> pd.DataFrame:
    return pd.DataFrame({k: {m: v.get(m) for m in keys} for k, v in results.items()}).T
