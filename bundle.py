"""Export the browser bundle and the parity fixture.

bundle.json.gz carries everything the in browser engine needs: the phenotype
ontology (labels, parents, information content, background), the recogniser
lexicon, every disease with its Mondo concept, genes and inheritance flags,
the direct and NOT annotations as compressed sparse rows, frequency phrases,
the browser ranker and its settings.

parity_fixture.json holds Python outputs the JavaScript engine must reproduce:
browser ranker differentials for development cases, and extraction results
for rendered notes.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from .. import config
from ..extract.recognizer import Recognizer
from ..extract.render import render
from ..knowledge.build import load_tables
from ..rank.engine import ADULT_ONSET, X_RECESSIVE, Engine, load_params
from ..rank.features import BROWSER_SHORTLIST, feature_frame, shortlist
from ..rank.fusion import load_browser

N_PARITY_CASES = 300
N_PARITY_NOTES = 150


def r4(x) -> list:
    return [round(float(v), 6) for v in x]


def build(eng: Engine, rec: Recognizer) -> dict:
    onto, prof = eng.onto, eng.prof
    t = eng.t
    terms = {
        "ids": onto.ids, "labels": onto.labels, "parents": onto.parents,
        "ic": r4(eng.ic), "bg": [float(f"{v:.8g}") for v in eng.bg],
        "pheno": [int(x) for x in onto.phenotype_mask],
        "system": [onto.index.get(s, -1) for s in onto.systems],
    }
    lexicon = []
    for (phrase, tid), via in rec.kind.items():
        lexicon.append([phrase, onto.index[tid], 0 if via == "label" else 1])
    ann = t["annotations"]
    pheno = ann[ann.aspect.eq("P") & ~ann.negated & ann.hpo_id.isin(onto.index)]
    raw = pheno.sort_values("freq", ascending=False).drop_duplicates(["disease_id", "hpo_id"])
    raw_map = dict(zip(zip(raw.disease_id, raw.hpo_id), raw.frequency))
    vocab: dict[str, int] = {}
    DIR = prof.DIR
    freq_code = []
    for d in range(prof.n):
        did = prof.disease_ids[d]
        for j in DIR.indices[DIR.indptr[d]:DIR.indptr[d + 1]]:
            code = raw_map.get((did, onto.ids[j]), "")
            freq_code.append(vocab.setdefault(code, len(vocab)))
    d = t["diseases"].set_index("disease_id").reindex(prof.disease_ids)
    concept = [m if isinstance(m, str) and m else i for i, m in zip(prof.disease_ids, d["mondo_id"])]
    inh = [list(x) if isinstance(x, (list, np.ndarray)) else [] for x in d["inheritance"]]
    onset = [list(x) if isinstance(x, (list, np.ndarray)) else [] for x in d["onset"]]
    model, T = load_browser()
    model_json = json.loads((config.MODELS / "fusion_browser.json").read_text())
    p = eng.scorer.params
    return {
        "version": 1,
        "manifest": json.loads((config.PROCESSED / "MANIFEST.json").read_text()) | {"tables": None},
        "terms": terms,
        "lexicon": lexicon,
        "diseases": {
            "ids": prof.disease_ids, "names": d["name"].fillna("").tolist(), "concept": concept,
            "genes": [list(x) if isinstance(x, (list, np.ndarray)) else [] for x in d["genes"]],
            "inheritance": inh,
            "xr": [int(X_RECESSIVE in x) for x in inh],
            "adult_only": [int(len(o) > 0 and set(o) <= ADULT_ONSET) for o in onset],
        },
        "dir": {"indptr": DIR.indptr.tolist(), "indices": DIR.indices.tolist(), "data": r4(DIR.data),
                "freq_code": freq_code, "freq_vocab": [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]},
        "not": {"indptr": prof.NOT.indptr.tolist(), "indices": prof.NOT.indices.tolist()},
        "model": model_json,
        "params": {"lam": p.lam, "eta": p.eta, "not_floor": p.not_floor, "exc_floor": p.exc_floor,
                   "shortlist": BROWSER_SHORTLIST},
    }


def browser_differential(eng: Engine, model, T: float, concept_of: dict, case: dict) -> dict:
    pat = eng.patient(case["observed"], case["excluded"], case.get("sex", "UNKNOWN_SEX"), case.get("age_years"))
    sig = eng.score(pat)
    f = feature_frame(eng, sig, idx=shortlist(sig, BROWSER_SHORTLIST))
    f["score"] = model.score(f)
    z = np.exp((f.score - f.score.max()) / T)
    f["prob"] = z / z.sum()
    f["concept"] = f.disease_id.map(concept_of)
    conc = f.groupby("concept").agg(prob=("prob", "sum"), score=("score", "max")).sort_values(["prob", "score"], ascending=False)
    gold = eng.row.get(case.get("disease_id"))
    return {
        "observed_used": [eng.onto.ids[i] for i in pat.observed],
        "excluded_used": [eng.onto.ids[i] for i in pat.excluded],
        "top_concepts": conc.index[:10].tolist(),
        "top_probs": r4(conc.prob[:10]),
        "shortlist_size": int(len(f)),
        "gold_lr": float(sig["lr"][gold]) if gold is not None else None,
        "gold_bma": float(sig["bma"][gold]) if gold is not None else None,
    }


def run() -> dict:
    config.ensure_dirs()
    t = load_tables()
    eng = Engine(t, params=load_params())
    rec = Recognizer(eng.onto, eng.ic)
    bundle = build(eng, rec)
    out_dir = config.WEB / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(bundle, separators=(",", ":")).encode()
    (out_dir / "bundle.json.gz").write_bytes(gzip.compress(raw, 9))

    model, T = load_browser()
    concept_of = dict(zip(eng.prof.disease_ids, bundle["diseases"]["concept"]))
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    dev = split[split.split.eq("dev")].sample(n=N_PARITY_CASES, random_state=config.SEED + 5)
    cases = []
    for c in dev.itertuples():
        case = {"case_id": c.case_id, "observed": list(c.observed), "excluded": list(c.excluded), "sex": c.sex,
                "age_years": None if pd.isna(c.age_years) else float(c.age_years), "disease_id": c.disease_id}
        case["expected"] = browser_differential(eng, model, T, concept_of, case)
        cases.append(case)
    notes = []
    pool = [eng.onto.ids[i] for i in range(eng.onto.n) if eng.onto.phenotype_mask[i] and 3 <= eng.ic[i] <= 6][::7]
    for k, c in enumerate(dev.head(N_PARITY_NOTES).itertuples()):
        note = render(eng.onto, list(c.observed), list(c.excluded), c.sex, c.age_years, seed=k, distractor_pool=pool)
        found = rec.extract(note["text"])
        notes.append({"text": note["text"], "findings": [[f.hpo_id, f.start, f.end, f.status] for f in found]})
    fixture = {"cases": cases, "notes": notes}
    (out_dir / "parity_fixture.json").write_text(json.dumps(fixture))
    return {"bundle_bytes_raw": len(raw), "bundle_bytes_gz": (out_dir / "bundle.json.gz").stat().st_size,
            "parity_cases": len(cases), "parity_notes": len(notes)}
