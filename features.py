"""Shortlist and features for the fused ranker.

The shortlist is the union of the top candidates of each base ranker, so a
disease only needs to look plausible to one view to be considered. Features
are the raw signals, their gap to the best value for this patient, and the log
rank under each base ranker. All features are cheap to compute in a browser,
which keeps the linear product model portable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Engine, Signals

BASE = ["lr", "lr_obs", "lr_exc", "support", "against", "not_conflicts", "exc_against",
        "p2d", "d2p", "bma", "exact", "bm25", "lsa", "size", "xr_female", "adult_onset_child"]
GAPS = ["lr", "bma", "bm25", "lsa"]
RANKS = ["lr", "bma", "bm25", "lsa", "d2p"]
FEATURES = BASE + [f"gap_{k}" for k in GAPS] + [f"logrank_{k}" for k in RANKS]

# Direction each feature may push a disease's score: +1 can only raise it, -1 can only lower it,
# 0 is free. Evidence keeps its clinical meaning: a recorded absence of a feature the disease
# usually shows, a NOT conflict or a worse rank can never count in the disease's favour. Without
# these constraints the rankers learn a curation artifact of published cases (excluded features
# are disproportionately the true disease's own hallmark features) and reverse that meaning.
MONOTONE = {
    "lr": 1, "lr_obs": 1, "lr_exc": 1, "support": 1, "against": -1, "not_conflicts": -1, "exc_against": -1,
    "p2d": 1, "d2p": 1, "bma": 1, "exact": 1, "bm25": 1, "lsa": 1, "size": 0, "xr_female": -1,
    "adult_onset_child": -1, "gap_lr": 1, "gap_bma": 1, "gap_bm25": 1, "gap_lsa": 1,
    "logrank_lr": -1, "logrank_bma": -1, "logrank_bm25": -1, "logrank_lsa": -1, "logrank_d2p": -1,
}

SHORTLIST = {"lr": 120, "bma": 80, "d2p": 40, "bm25": 50, "lsa": 30}

# The browser engine has no text indexes: its ranker uses only ontology features
# and shortlists by the ontology rankers alone.
BROWSER_FEATURES = [f for f in FEATURES if "bm25" not in f and "lsa" not in f]
BROWSER_SHORTLIST = {"lr": 120, "bma": 80, "d2p": 40}


def ranks(scores: np.ndarray) -> np.ndarray:
    """Competition rank, 1 for the best, ties share the better rank."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    r = np.empty(len(scores), dtype=np.int64)
    first = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    pos = np.arange(1, len(scores) + 1)
    group_rank = np.maximum.accumulate(np.where(first, pos, 0))
    r[order] = group_rank
    return r


def rounded(scores: np.ndarray) -> np.ndarray:
    return np.round(scores.astype(np.float64), 3)


def mid_rank(scores: np.ndarray, i: int) -> float:
    """Rank of item i with ties split evenly: the expected rank under random tie breaking.
    Evaluation uses this so that a coarse score cannot win by tying with many others."""
    v = scores[i]
    return float((scores > v).sum() + ((scores == v).sum() - 1) / 2.0 + 1)


def shortlist(sig: Signals, spec: dict[str, int] | None = None) -> np.ndarray:
    pick = set()
    for key, k in (spec or SHORTLIST).items():
        s = rounded(sig[key])
        # Stable order on rounded values: ties are broken by disease index, so every implementation agrees.
        top = np.argsort(-s, kind="stable")[:k]
        pick.update(top.tolist())
    return np.array(sorted(pick), dtype=np.int64)


def feature_frame(eng: Engine, sig: Signals, idx: np.ndarray | None = None) -> pd.DataFrame:
    idx = shortlist(sig) if idx is None else idx
    data = {k: sig[k][idx] for k in BASE}
    for k in GAPS:
        data[f"gap_{k}"] = sig[k][idx] - sig[k].max()
    for k in RANKS:
        # Rank on values rounded to three decimals, so diseases with identical profiles always
        # share a rank whatever floating point order produced their scores.
        data[f"logrank_{k}"] = np.log(ranks(rounded(sig[k]))[idx].astype(np.float64))
    df = pd.DataFrame(data)
    df.insert(0, "row", idx)
    df.insert(1, "disease_id", [eng.prof.disease_ids[i] for i in idx])
    return df


def in_browser_shortlist(df: pd.DataFrame) -> np.ndarray:
    """Rows of a feature frame that the browser shortlist would also contain."""
    m = np.zeros(len(df), dtype=bool)
    for key, k in BROWSER_SHORTLIST.items():
        m |= df[f"logrank_{key}"].to_numpy() <= np.log(k) + 1e-9
    return m
