"""Teaching cases for the interface, chosen from the blind set after its single scoring.

The selection reads the stored blind predictions; it does not score the blind
set again. It deliberately includes successes, near misses and failures:
six cases ranked first, three ranked second to tenth, two ranked eleventh to
fiftieth and one beyond, each from a different disease. Each case is run
through the full copilot under the strict protocol with a question, and the
whole result is stored for the interface.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from ..agent.audit import AuditLog
from ..agent.copilot import Copilot, Request

BANDS = [((1, 1), 6), ((2, 10), 3), ((11, 50), 2), ((51, 10**9), 1)]
QUESTIONS = ["Which genes are linked to the leading candidate?", "How is the leading candidate inherited?",
             "What is meant by {feature}?"]


def _clean(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def describe(age, sex) -> str:
    who = {"FEMALE": ("girl", "woman"), "MALE": ("boy", "man")}.get(sex, ("child", "adult"))
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return {"FEMALE": "female", "MALE": "male"}.get(sex, "sex not recorded") + ", age not recorded"
    if age < 1:
        return f"{who[0]} under one year"
    if age < 18:
        return f"{who[0]} aged {int(age)}"
    return f"{who[1]} aged {int(age)}"


def select(pred: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    df = pred.merge(split[["case_id", "pmid", "observed", "excluded", "genes", "disease_label"]], on="case_id")
    df = df[df.n_observed.between(5, 14) & df.gold_profile_after.ge(5) & df.pmid.notna()]
    rng = np.random.default_rng(config.SEED + 11)
    picked, used = [], set()
    for (lo, hi), n in BANDS:
        pool = df[df.rank_concept.between(lo, hi)]
        order = rng.permutation(len(pool))
        for i in order:
            r = pool.iloc[i]
            if r.disease_id in used:
                continue
            used.add(r.disease_id)
            picked.append(r)
            if sum(1 for p in picked if lo <= p.rank_concept <= hi) >= n:
                break
    return pd.DataFrame(picked).reset_index(drop=True)


def run() -> str:
    pred = pd.read_parquet(config.REPORTS / "blind_predictions.parquet")
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    chosen = select(pred, split)
    # Successes first, then near misses, then failures, so the interface opens on a typical case.
    chosen = chosen.sort_values(["rank_concept", "case_id"]).reset_index(drop=True)
    cop = Copilot(audit=AuditLog(config.AUDIT / "teaching_log.jsonl"))
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet").set_index("disease_id")
    out = []
    for i, c in enumerate(chosen.itertuples()):
        age = None if pd.isna(c.age_years) else float(c.age_years)
        q = QUESTIONS[i % len(QUESTIONS)]
        if "{feature}" in q:
            first = next((t for t in c.observed if t in cop.engine.onto.index), None)
            q = q.format(feature=cop.engine.onto.labels[cop.engine.onto.index[first]].lower())
        res = cop.run(Request(observed=list(c.observed), excluded=list(c.excluded), sex=c.sex, age_years=age,
                              strict_pmid=c.pmid, question=q))
        cited = {k for s in res["sentences"] for k in s["cites"]}
        res["sources"] = {k: v for k, v in res["sources"].items() if k in cited}
        res.pop("findings", None)
        concept = diseases.at[c.disease_id, "mondo_id"] if c.disease_id in diseases.index else None
        out.append({
            "title": f"Case {i + 1:02d}: {describe(age, c.sex)}, {len(res['patient']['observed'])} features",
            "case_id": c.case_id, "pmid": c.pmid, "sex": c.sex, "age_years": age, "question": q,
            "rank_concept": float(c.rank_concept),
            "gold": {"disease_id": c.disease_id, "label": c.disease_label,
                     "concept": concept if isinstance(concept, str) else c.disease_id,
                     "gene": (list(c.genes) or [None])[0]},
            "result": _clean(res),
        })
    path = config.REPORTS / "teaching_cases.json"
    path.write_text(json.dumps(out))
    return f"{len(out)} teaching cases written to {path}"
