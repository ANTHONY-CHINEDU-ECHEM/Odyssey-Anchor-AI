"""Extraction evaluation on rendered notes.

Each case is rendered as a note (see extract/render.py) and read back. Scores:

  observed   precision and recall of present features, exact identifier
  tolerant   precision counting a prediction correct when it is an ancestor or
             descendant of a gold feature (a coarser but true reading)
  excluded   precision and recall of absent features
  family     share of family history distractors wrongly read as the patient's
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..extract.recognizer import Recognizer
from ..extract.render import render
from ..ontology.graph import Ontology


def run(cases: pd.DataFrame, onto: Ontology, rec: Recognizer, ic) -> dict:
    distractors = [onto.ids[i] for i in range(onto.n) if onto.phenotype_mask[i] and 3.0 <= ic[i] <= 6.0][::7]
    tp = fp = fn = etp = efp = efn = fam_leak = fam_total = tol_tp = 0
    examples = []
    for k, c in enumerate(cases.itertuples()):
        note = render(onto, list(c.observed), list(c.excluded), c.sex, c.age_years, seed=config.SEED + k,
                      distractor_pool=distractors)
        out = rec.to_terms(rec.extract(note["text"]))
        go, po = set(note["observed"]), set(out["observed"])
        ge, pe = set(note["excluded"]), set(out["excluded"])
        tp += len(go & po)
        fp += len(po - go)
        fn += len(go - po)
        etp += len(ge & pe)
        efp += len(pe - ge)
        efn += len(ge - pe)
        fam = set(note["family"])
        fam_total += len(fam)
        fam_leak += len(fam & po)
        gold_idx = [onto.index[t] for t in go if t in onto.index]
        for p in po:
            if p in go:
                tol_tp += 1
                continue
            i = onto.index.get(p)
            if i is None:
                continue
            if any(onto.is_ancestor(i, g) or onto.is_ancestor(g, i) for g in gold_idx):
                tol_tp += 1
        if len(examples) < 6 and (po - go or go - po):
            examples.append({"case_id": c.case_id, "note": note["text"][:700],
                             "missed": sorted(go - po)[:5], "extra": sorted(po - go)[:5]})
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return {
        "cases": int(len(cases)),
        "observed": {"precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9), "gold": tp + fn},
        "observed_tolerant_precision": tol_tp / max(tp + fp, 1),
        "excluded": {"precision": etp / max(etp + efp, 1), "recall": etp / max(etp + efn, 1), "gold": etp + efn},
        "family_leak_rate": fam_leak / max(fam_total, 1), "family_distractors": fam_total,
        "examples": examples,
    }
