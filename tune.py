"""Choose the likelihood ratio settings on development cases only, under the strict protocol."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .. import config
from ..ontology import profiles as P
from .engine import Engine, save_params
from .scorers import LRParams, Patient

GRID_LAMBDA = (0.02, 0.05, 0.1, 0.2, 0.4)
GRID_ETA = (0.005, 0.02, 0.05)


def run(n_cases: int = 800) -> dict:
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    dev = split[split.split.eq("dev")].sample(n=n_cases, random_state=config.SEED)
    eng = Engine()
    prepared = []
    for c in dev.itertuples():
        pat = Patient.from_terms(eng.onto, c.observed, c.excluded, c.sex, c.age_years)
        sp_ = P.strict_profiles(eng.strict, c.pmid, eng.onto)
        prepared.append((pat, sp_, eng.row[c.disease_id]))
    results = []
    for lam, eta in itertools.product(GRID_LAMBDA, GRID_ETA):
        eng.scorer.params = LRParams(lam=lam, eta=eta)
        ranks = []
        for pat, sp_, g in prepared:
            s = eng.scorer.likelihood_ratio(pat, eng.prof)["lr"]
            if sp_ is not None:
                s[[eng.row[d] for d in sp_.disease_ids]] = eng.scorer.likelihood_ratio(pat, sp_)["lr"]
            ranks.append(int((s > s[g]).sum()) + 1)
        ranks = np.array(ranks)
        results.append({"lam": lam, "eta": eta, "mrr": float((1 / ranks).mean()),
                        "top1": float((ranks <= 1).mean()), "top10": float((ranks <= 10).mean())})
        print(results[-1])
    best = max(results, key=lambda r: r["mrr"])
    save_params(LRParams(lam=best["lam"], eta=best["eta"]))
    out = {"n_dev_cases": n_cases, "criterion": "mean reciprocal rank, strict protocol", "grid": results, "chosen": best}
    (config.REPORTS / "lr_tuning.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    config.ensure_dirs()
    print(json.dumps(run()["chosen"], indent=2))
