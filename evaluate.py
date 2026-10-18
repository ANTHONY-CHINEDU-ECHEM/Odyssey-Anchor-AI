"""Evaluation orchestration: development as often as needed, the blind set once.

make evaluate         development evaluation, writes reports/evaluation_dev.json
make evaluate_blind   checks the frozen hash, scores the blind set, writes
                      reports/evaluation_blind.json and the blind predictions,
                      and appends a timestamped entry to data/eval/blind_runs.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..agent.audit import AuditLog
from ..agent.copilot import Copilot, Request
from . import extraction_eval, freeze, nbq_eval, probes, retrieval_eval, verifier_eval
from .metrics import bootstrap_ci, calibration, topk
from .perturb import TRANSFORMS
from .runner import rank_cases

AGE_BANDS = [(-1, 1, "under 1"), (1, 6, "1 to 5"), (6, 18, "6 to 17"), (18, 200, "18 and over")]
OBS_BANDS = [(3, 6, "3 to 5"), (6, 11, "6 to 10"), (11, 21, "11 to 20"), (21, 10_000, "over 20")]
PROFILE_BANDS = [(0, 1, "none left"), (1, 10, "1 to 9"), (10, 30, "10 to 29"), (30, 100_000, "30 or more")]


def band(v, bands, missing="unknown"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return missing
    for lo, hi, name in bands:
        if lo <= v < hi:
            return name
    return missing


def inheritance_group(inh) -> str:
    inh = set(inh) if inh is not None else set()
    if any("X-linked" in x for x in inh):
        return "X linked"
    if "Autosomal dominant inheritance" in inh and "Autosomal recessive inheritance" not in inh:
        return "autosomal dominant"
    if "Autosomal recessive inheritance" in inh and "Autosomal dominant inheritance" not in inh:
        return "autosomal recessive"
    if inh:
        return "mixed or other"
    return "not stated"


def enrich(pred: pd.DataFrame, split: pd.DataFrame, diseases: pd.DataFrame) -> pd.DataFrame:
    d = diseases.set_index("disease_id")
    df = pred.merge(split[["case_id", "cohort", "group", "disease_id", "sex", "age_years", "n_observed", "n_excluded"]],
                    on="case_id", how="left")
    df["age_band"] = df.age_years.map(lambda v: band(v, AGE_BANDS))
    df["obs_band"] = df.n_observed.map(lambda v: band(v, OBS_BANDS))
    df["profile_band"] = df.gold_profile_after.map(lambda v: band(v, PROFILE_BANDS))
    df["inheritance"] = df.disease_id.map(lambda x: inheritance_group(d.at[x, "inheritance"] if x in d.index else None))
    df["sex_group"] = df.sex.map({"MALE": "male", "FEMALE": "female"}).fillna("not recorded")
    return df


def subgroup_table(df: pd.DataFrame) -> dict:
    out = {}
    for col in ("sex_group", "age_band", "obs_band", "profile_band", "inheritance"):
        rows = {}
        for k, g in df.groupby(col):
            if len(g) < 20:
                continue
            m = topk(g.rank_concept)
            rows[k] = {"n": int(len(g)), "top1": m["top1"], "top10": m["top10"], "mrr": m["mrr"]}
        out[col] = rows
    return out


def headline(df: pd.DataFrame) -> dict:
    out = {"identifier": topk(df.rank_id), "concept": topk(df.rank_concept)}
    for k in ("top1", "top10"):
        kk = int(k[3:])
        out["concept"][f"{k}_ci95"] = bootstrap_ci((df.rank_concept <= kk).astype(float), groups=df.group)
    out["concept"]["mrr_ci95"] = bootstrap_ci(1.0 / df.rank_concept, groups=df.group)
    out["in_shortlist"] = float(df.in_shortlist.mean())
    if "rank_browser_concept" in df:
        out["browser_concept"] = topk(df.rank_browser_concept)
    base = {}
    for k in ("lr", "bma", "p2d", "d2p", "exact", "bm25", "lsa"):
        base[k] = topk(df[f"rank_{k}"])
    out["base_rankers_identifier"] = base
    g = df.dropna(subset=["rank_gene"])
    out["gene"] = topk(g.rank_gene.astype(float))
    out["calibration_top1"] = calibration(df.top_prob, (df.rank_concept == 1).astype(float))
    diag = df[df.gold_profile_after >= 3]
    out["diagnosable_subset"] = {"share": float(len(diag) / len(df)), "concept": topk(diag.rank_concept)}
    return out


def copilot_runs(cop: Copilot, cases: pd.DataFrame, n: int, questions: list[str]) -> list[dict]:
    sample = cases.sample(n=min(n, len(cases)), random_state=config.SEED)
    out = []
    onto = cop.engine.onto
    for i, c in enumerate(sample.itertuples()):
        q = questions[i % len(questions)]
        if "{feature}" in q:
            first = next((t for t in c.observed if t in onto.index), None)
            q = q.format(feature=onto.labels[onto.index[first]].lower()) if first else ""
        res = cop.run(Request(observed=list(c.observed), excluded=list(c.excluded), sex=c.sex,
                              age_years=None if pd.isna(c.age_years) else float(c.age_years),
                              strict_pmid=c.pmid, question=q, log=False))
        out.append(res)
    return out


ASK = ["Which genes are linked to the leading candidate?", "How is the leading candidate inherited?",
       "What is meant by {feature}?", ""]


def copilot_quality(runs: list[dict]) -> dict:
    total = sum(r["verification"]["total"] for r in runs)
    passed = sum(r["verification"]["passed"] for r in runs)
    reasons = {}
    for r in runs:
        for d in r["verification"]["dropped"]:
            for x in d["reasons"]:
                reasons[x] = reasons.get(x, 0) + 1
    timings = pd.DataFrame([r["timings_ms"] for r in runs])
    return {"runs": len(runs), "sentences": total, "verified": passed, "verified_share": passed / max(total, 1),
            "drop_reasons": reasons, "median_ms_by_node": timings.median().round(2).to_dict(),
            "median_total_ms": float(timings.sum(axis=1).median())}


def evaluate_dev(fast: bool = False) -> dict:
    config.ensure_dirs()
    t0 = time.time()
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet")
    dev = split[split.split.eq("dev")]
    meta = json.loads((config.MODELS / "fusion_meta.json").read_text())
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "fusion_cross_validation": meta["cv"], "product": meta["product"],
              "temperature": meta["temperature"], "base_rankers_dev": meta["base_rankers_dev"]}
    # Out of fold product predictions for subgroups and calibration on development data.
    sub = dev.sample(n=600 if fast else 1500, random_state=config.SEED)
    strict_pred = enrich(rank_cases(sub, strict=True, label="dev strict"), split, diseases)
    released = rank_cases(sub, strict=False, label="dev released")
    report["leakage_dev"] = {"strict": topk(strict_pred.rank_concept), "released": topk(released.rank_concept),
                             "note": "same cases; final product model trained on development data, so these are in sample"}
    report["subgroups_dev_in_sample"] = subgroup_table(strict_pred)
    cop = Copilot(audit=AuditLog(config.AUDIT / "evaluation_dev_log.jsonl"))
    report["extraction_dev"] = extraction_eval.run(dev.sample(n=500, random_state=config.SEED), cop.engine.onto,
                                                    cop.recognizer, cop.engine.ic)
    q = pd.read_json(config.EVAL / "questions.jsonl", lines=True)
    report["retrieval_dev"] = retrieval_eval.run(cop.passages, q[q.split.eq("dev")])
    runs = copilot_runs(cop, dev, 60 if fast else 150, ASK)
    report["copilot_dev"] = copilot_quality(runs)
    report["verifier_dev"] = verifier_eval.run(runs, cop.gene_set)
    report["probes_dev"] = probes.run(cop)
    report["next_question_dev"] = nbq_eval.run(dev, n=60 if fast else 150)
    report["seconds"] = round(time.time() - t0, 1)
    (config.REPORTS / "evaluation_dev.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def evaluate_blind() -> dict:
    config.ensure_dirs()
    frozen = freeze.verify()
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet")
    blind = split[split.split.eq("blind")]
    report = {"started_utc": started, "frozen": {k: frozen[k] for k in ("frozen_utc", "blind_case_sha256", "blind_question_sha256")}}
    pred = enrich(rank_cases(blind, strict=True, label="blind strict", with_browser=True), split, diseases)
    pred.to_parquet(config.REPORTS / "blind_predictions.parquet", index=False)
    report["headline"] = headline(pred)
    report["subgroups"] = subgroup_table(pred)
    released = rank_cases(blind, strict=False, label="blind released")
    report["leakage"] = {"strict": topk(pred.rank_concept), "released": topk(released.rank_concept),
                         "strict_identifier": topk(pred.rank_id), "released_identifier": topk(released.rank_id)}
    robustness = {"unchanged": topk(pred.rank_concept)}
    for name, tf in TRANSFORMS.items():
        r = rank_cases(blind, strict=True, transform=tf, label=f"blind {name}")
        robustness[name] = {**topk(r.rank_concept), "description": tf.name, "cases_scored": int(len(r))}
    report["robustness"] = robustness
    cop = Copilot(audit=AuditLog(config.AUDIT / "evaluation_blind_log.jsonl"))
    report["extraction"] = extraction_eval.run(blind, cop.engine.onto, cop.recognizer, cop.engine.ic)
    q = pd.read_json(config.EVAL / "questions.jsonl", lines=True)
    rq = retrieval_eval.run(cop.passages, q[q.split.eq("blind")])
    report["retrieval"] = {k: v for k, v in rq.items() if k != "misses"}
    report["retrieval_misses"] = rq["misses"]
    runs = copilot_runs(cop, blind, 200, ASK)
    report["copilot"] = copilot_quality(runs)
    report["verifier"] = verifier_eval.run(runs, cop.gene_set)
    report["probes"] = probes.run(cop)
    report["next_question"] = nbq_eval.run(blind, n=200)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["seconds"] = round(time.time() - t0, 1)
    (config.REPORTS / "evaluation_blind.json").write_text(json.dumps(report, indent=2, default=float))
    runs_path = config.EVAL / "blind_runs.json"
    log = json.loads(runs_path.read_text()) if runs_path.exists() else {"runs": []}
    log["runs"].append({"started_utc": started, "finished_utc": report["finished_utc"],
                        "blind_case_sha256": frozen["blind_case_sha256"],
                        "concept_top1": report["headline"]["concept"]["top1"],
                        "concept_top10": report["headline"]["concept"]["top10"]})
    runs_path.write_text(json.dumps(log, indent=2))
    return report


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if which == "blind":
        rep = evaluate_blind()
        print(json.dumps(rep["headline"]["concept"], indent=2))
    else:
        rep = evaluate_dev(fast="--fast" in sys.argv)
        print(json.dumps({k: rep[k] for k in ("product", "fusion_cross_validation", "leakage_dev")}, indent=2))
