"""Parallel scoring of cases with the frozen product model.

Each worker holds an engine, the product ranker, the Mondo concept map and the
gene index. `rank_cases` returns one row per case: product rank by identifier
and by concept, every base ranker's rank, the gene rank, the top concept and
its probability, and whether the true disease reached the shortlist.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import time
from typing import Callable

import numpy as np
import pandas as pd

from .. import config
from ..rank.engine import Engine, load_params
from ..rank.features import BROWSER_SHORTLIST, feature_frame, mid_rank, shortlist
from ..rank.fusion import load_browser, load_product
from ..reason.genes import GeneIndex

_W: dict = {}
BASE = ["lr", "bma", "p2d", "d2p", "exact", "bm25", "lsa"]


def _init(strict: bool, with_browser: bool = False) -> None:
    eng = Engine(params=load_params())
    model, meta = load_product()
    _W["browser"] = None
    if with_browser:
        try:
            _W["browser"] = load_browser()
        except FileNotFoundError:
            pass
    _W.update({
        "eng": eng, "model": model, "T": float(meta["temperature"]), "strict": strict,
        "concept": {d: (m if isinstance(m, str) else d) for d, m in zip(eng.t["diseases"].disease_id, eng.t["diseases"].mondo_id)},
        "genes": GeneIndex(eng.t["disease_genes"]),
    })


def product_rank(case: dict) -> dict:
    eng, model, T = _W["eng"], _W["model"], _W["T"]
    pat = eng.patient(case["observed"], case["excluded"], case.get("sex", "UNKNOWN_SEX"), case.get("age_years"))
    sig = eng.score(pat, pmid=case.get("pmid"), strict=_W["strict"])
    g = eng.row[case["disease_id"]]
    feats = feature_frame(eng, sig)
    in_short = bool((feats.row == g).any())
    feats["score"] = model.score(feats)
    z = np.exp((feats.score - feats.score.max()) / T)
    feats["prob"] = z / z.sum()
    feats["concept"] = feats.disease_id.map(_W["concept"])
    gold_concept = _W["concept"].get(case["disease_id"], case["disease_id"])
    base = {f"rank_{k}": mid_rank(sig[k], g) for k in BASE}
    conc = feats.groupby("concept").agg(score=("score", "max"), prob=("prob", "sum")).sort_values(["prob", "score"], ascending=False)
    top_concept = conc.index[0]
    prof, prow = eng.profile_row(case["disease_id"], sig)
    rec = {"case_id": case["case_id"], "in_shortlist": in_short, "n_observed_used": int(len(pat.observed)),
           "gold_profile_after": int(prof.DIR.indptr[prow + 1] - prof.DIR.indptr[prow]),
           "n_excluded_used": int(len(pat.excluded)), "top_concept": top_concept,
           "top_prob": float(conc.prob.iloc[0]), "top_disease": feats.sort_values("score").disease_id.iloc[-1], **base}
    if in_short:
        sc = feats.score.to_numpy()
        rec["rank_id"] = mid_rank(sc, int(np.flatnonzero(feats.row.to_numpy() == g)[0]))
        cs = conc.prob.to_numpy()
        rec["rank_concept"] = mid_rank(cs, int(conc.index.get_loc(gold_concept)))
        rec["gold_prob"] = float(conc.loc[gold_concept, "prob"])
    else:
        rec["rank_id"] = rec["rank_concept"] = int(len(feats) + base["rank_lr"])
        rec["gold_prob"] = 0.0
    if _W.get("browser") is not None:
        bmodel = _W["browser"][0]
        bf = feature_frame(eng, sig, idx=shortlist(sig, BROWSER_SHORTLIST))
        bf["score"] = bmodel.score(bf)
        bf["concept"] = bf.disease_id.map(_W["concept"])
        if (bf.row == g).any():
            bc = bf.groupby("concept").score.max()
            rec["rank_browser_concept"] = mid_rank(bc.to_numpy(), int(bc.index.get_loc(gold_concept)))
        else:
            rec["rank_browser_concept"] = int(len(bf) + base["rank_lr"])
    gene_best = _W["genes"].gene_scores(feats.disease_id.tolist(), feats.score.tolist())
    gold_genes = [x for x in case.get("genes", []) if x]
    if gold_genes and gene_best:
        ordered = sorted(gene_best.values(), reverse=True)
        best = max((gene_best.get(x, -np.inf) for x in gold_genes))
        rec["rank_gene"] = int(np.searchsorted(-np.array(ordered), -best, side="left") + 1) if np.isfinite(best) else 10_000
    else:
        rec["rank_gene"] = None
    return rec


def _job(args):
    case, transform, seed = args
    if transform is not None:
        case = transform(dict(case), random.Random(seed))
        if case is None:
            return None
    return product_rank(case)


def records(cases: pd.DataFrame) -> list[dict]:
    recs = cases[["case_id", "observed", "excluded", "sex", "age_years", "pmid", "disease_id", "genes"]].to_dict("records")
    for r in recs:
        r["observed"], r["excluded"], r["genes"] = list(r["observed"]), list(r["excluded"]), list(r["genes"])
        r["age_years"] = None if pd.isna(r["age_years"]) else float(r["age_years"])
    return recs


def rank_cases(cases: pd.DataFrame, strict: bool = True, transform: Callable | None = None,
               workers: int | None = None, label: str = "", with_browser: bool = False) -> pd.DataFrame:
    recs = records(cases)
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    t0 = time.time()
    out = []
    jobs = [(r, transform, config.SEED + i) for i, r in enumerate(recs)]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(strict, with_browser)) as pool:
        for i, rec in enumerate(pool.imap(_job, jobs, chunksize=8)):
            if rec is not None:
                out.append(rec)
            if (i + 1) % 500 == 0:
                print(f"{label}: {i + 1}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(out)
