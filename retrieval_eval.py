"""Knowledge retrieval evaluation on the frozen questions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..retrieval.passages import PassageIndex, contextual_variants

METHODS = ["bm25", "lsa", "hybrid", "learned", "learned_without_kind_fit",
           "learned_without_ent_disease", "learned_without_lsa"]


def rank_of_gold(df: pd.DataFrame, gold: set[str]) -> int:
    hit = np.flatnonzero(df.pid.isin(gold).to_numpy())
    return int(hit[0]) + 1 if len(hit) else 10_000


def run(idx: PassageIndex, questions: pd.DataFrame) -> dict:
    ranks = {m: [] for m in METHODS}
    misses = []
    variants = pd.DataFrame(contextual_variants(questions, sorted(idx.by_disease), seed=99))
    questions = variants
    for q in variants.itertuples():
        df, info = idx.candidates(q.question, q.context)
        gold = set(q.gold)
        for m in METHODS:
            s = idx.score_frame(df, m)
            order = df.iloc[np.argsort(-s, kind="stable")]
            ranks[m].append(rank_of_gold(order, gold))
            if m == "learned" and ranks[m][-1] > 1:
                misses.append({"qid": q.qid, "kind": q.kind, "question": q.question, "gold": list(gold),
                               "got": order.pid.head(3).tolist(), "rank": ranks[m][-1], "intent": info["intent"]})
    out = {}
    for m, r in ranks.items():
        r = np.array(r)
        out[m] = {"hit1": float((r <= 1).mean()), "recall3": float((r <= 3).mean()), "mrr": float((1 / r).mean()),
                  "ndcg10": float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean())}
    by_kind = {}
    learned = np.array(ranks["learned"])
    for kind in sorted(questions.kind.unique()):
        m = (questions.kind == kind).to_numpy()
        by_kind[kind] = {"n": int(m.sum()), "hit1": float((learned[m] <= 1).mean()),
                         "mrr": float((1 / learned[m]).mean())}
    return {"n": int(len(questions)), "methods": out, "by_kind": by_kind, "misses": misses}
