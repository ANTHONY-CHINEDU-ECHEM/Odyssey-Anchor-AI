"""Score cases with every base ranker and store base ranks and shortlist features.

Runs in parallel worker processes, each holding its own engine. The output for
a split is one parquet of shortlist features (with the label) and one parquet
of base ranker ranks per case, so evaluation never needs to rescore.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from .. import config
from .engine import Engine, load_params
from .features import FEATURES, feature_frame, ranks

_ENG: Engine | None = None
BASE_RANKERS = ["lr", "bma", "p2d", "d2p", "exact", "bm25", "lsa"]


def _init() -> None:
    global _ENG
    _ENG = Engine(params=load_params())


def score_case(case: dict, strict: bool = True) -> tuple[pd.DataFrame, dict]:
    eng = _ENG
    pat = eng.patient(case["observed"], case["excluded"], case["sex"], case["age_years"])
    sig = eng.score(pat, pmid=case["pmid"], strict=strict)
    g = eng.row[case["disease_id"]]
    feats = feature_frame(eng, sig)
    if g not in set(feats.row):
        extra = feature_frame(eng, sig, idx=np.array([g]))
        feats = pd.concat([feats, extra], ignore_index=True)
        in_shortlist = False
    else:
        in_shortlist = True
    feats["label"] = (feats.row == g).astype(np.int8)
    feats.insert(0, "case_id", case["case_id"])
    rec = {"case_id": case["case_id"], "in_shortlist": in_shortlist, "shortlist_size": int(len(feats) - (0 if in_shortlist else 1)),
           "n_strict_changed": len(sig.strict_diseases), "gold_size_released": int(eng.prof.size[g])}
    for k in BASE_RANKERS:
        rec[f"rank_{k}"] = int(ranks(sig[k])[g])
    # Profile size of the gold disease after strict removal, read from the features.
    rec["gold_size_after"] = float(np.expm1(feats.loc[feats.label.eq(1), "size"].iloc[0]))
    return feats, rec


def _work(case: dict):
    return score_case(case, strict=True)


def run(split_name: str, limit: int | None = None, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    cases = split[split.split.eq(split_name)]
    if limit:
        cases = cases.sample(n=min(limit, len(cases)), random_state=config.SEED)
    records = cases[["case_id", "observed", "excluded", "sex", "age_years", "pmid", "disease_id"]].to_dict("records")
    for r in records:
        r["observed"], r["excluded"] = list(r["observed"]), list(r["excluded"])
        r["age_years"] = None if pd.isna(r["age_years"]) else float(r["age_years"])
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    t0 = time.time()
    feats, recs = [], []
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for i, (f, r) in enumerate(pool.imap(_work, records, chunksize=8)):
            feats.append(f)
            recs.append(r)
            if (i + 1) % 500 == 0:
                print(f"{split_name}: {i + 1}/{len(records)} cases, {time.time() - t0:.0f}s", flush=True)
    F = pd.concat(feats, ignore_index=True)
    F[FEATURES] = F[FEATURES].astype(np.float32)
    R = pd.DataFrame(recs)
    return F, R


if __name__ == "__main__":
    import sys
    config.ensure_dirs()
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if name == "blind":
        raise SystemExit("Blind features are produced only inside evaluate_blind")
    F, R = run(name)
    F.to_parquet(config.MODELS / f"features_{name}.parquet", index=False)
    R.to_parquet(config.MODELS / f"base_ranks_{name}.parquet", index=False)
    print(R.describe().T.to_string())
