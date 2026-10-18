"""How much of the knowledge base is drawn from the evaluation patients' own papers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config


def run() -> dict:
    ann = pd.read_parquet(config.PROCESSED / "annotations.parquet")
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    pheno = ann[ann.aspect.eq("P") & ~ann.negated]
    cited = set(pheno.refs.explode().dropna())
    single = pheno[pheno.refs.apply(len).eq(1)].assign(ref=lambda d: d.refs.str[0])
    by_dis = pheno.groupby("disease_id").size()
    only = single.groupby(["disease_id", "ref"]).size()
    removed_share, emptied = [], []
    for d, p in zip(split.disease_id, split.pmid):
        total = int(by_dis.get(d, 0))
        gone = int(only.get((d, p), 0)) if p else 0
        removed_share.append(gone / total if total else 0.0)
        emptied.append(total > 0 and gone == total)
    out = {
        "eligible_cases": int(len(split)),
        "cases_pmid_cited": float(split.pmid.isin(cited).mean()),
        "mean_share_of_true_disease_annotations_removed": float(np.mean(removed_share)),
        "cases_whose_true_disease_loses_every_annotation": float(np.mean(emptied)),
        "rule": "an annotation is removed for a case when its only reference is the case's own paper",
    }
    (config.REPORTS / "leakage_stats.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
