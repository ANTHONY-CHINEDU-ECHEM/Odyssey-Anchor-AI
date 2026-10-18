"""Shared fixtures: a small synthetic ontology and annotation table, plus the real data when present."""
from __future__ import annotations

import pandas as pd
import pytest

from odyssey_anchor import config
from odyssey_anchor.ontology import profiles as P
from odyssey_anchor.ontology.graph import Ontology

# HP:0000001 All
#   HP:0000118 Phenotypic abnormality
#     HP:0000707 Abnormality of the nervous system
#       HP:0001250 Seizure
#         HP:0007359 Focal seizure
#         HP:0002069 Bilateral tonic clonic seizure
#       HP:0001252 Hypotonia
#     HP:0000478 Abnormality of the eye
#       HP:0001083 Ectopia lentis
#       HP:0000545 Myopia
#   HP:0000005 Mode of inheritance
#     HP:0000006 Autosomal dominant inheritance
TERMS = [
    ("HP:0000001", "All", None), ("HP:0000118", "Phenotypic abnormality", "HP:0000001"),
    ("HP:0000707", "Abnormality of the nervous system", "HP:0000118"),
    ("HP:0001250", "Seizure", "HP:0000707"), ("HP:0007359", "Focal seizure", "HP:0001250"),
    ("HP:0002069", "Bilateral tonic clonic seizure", "HP:0001250"), ("HP:0001252", "Hypotonia", "HP:0000707"),
    ("HP:0000478", "Abnormality of the eye", "HP:0000118"), ("HP:0001083", "Ectopia lentis", "HP:0000478"),
    ("HP:0000545", "Myopia", "HP:0000478"), ("HP:0000005", "Mode of inheritance", "HP:0000001"),
    ("HP:0000006", "Autosomal dominant inheritance", "HP:0000005"),
]
SYSTEM = {"HP:0001250": "HP:0000707", "HP:0007359": "HP:0000707", "HP:0002069": "HP:0000707",
          "HP:0001252": "HP:0000707", "HP:0000707": "HP:0000707", "HP:0001083": "HP:0000478",
          "HP:0000545": "HP:0000478", "HP:0000478": "HP:0000478"}


@pytest.fixture(scope="session")
def tiny_tables():
    terms = pd.DataFrame({
        "id": [t[0] for t in TERMS], "label": [t[1] for t in TERMS],
        "definition": ["" for _ in TERMS], "definition_refs": [[] for _ in TERMS], "comment": "",
        "synonyms": [["Epileptic seizure", "Seizures"] if t[0] == "HP:0001250" else
                     (["Low muscle tone"] if t[0] == "HP:0001252" else []) for t in TERMS],
        "synonym_types": [[] for _ in TERMS], "xrefs": [[] for _ in TERMS], "obsolete": False,
        "replaced_by": None, "system": [SYSTEM.get(t[0], "") for t in TERMS],
    })
    edges = pd.DataFrame([{"child": t[0], "parent": t[2]} for t in TERMS if t[2]])
    rows = [
        # disease, term, freq, negated, refs
        ("OMIM:1", "HP:0007359", 0.9, False, ["PMID:111"]),
        ("OMIM:1", "HP:0001252", 0.5, False, ["OMIM:1"]),
        ("OMIM:2", "HP:0001083", 0.9, False, ["PMID:222"]),
        ("OMIM:2", "HP:0000545", 0.5, False, ["PMID:222", "OMIM:2"]),
        ("OMIM:2", "HP:0001250", 0.0, True, ["OMIM:2"]),
        ("OMIM:3", "HP:0001252", 0.9, False, ["PMID:333"]),
        ("OMIM:3", "HP:0002069", 0.2, False, ["PMID:333"]),
    ]
    ann = pd.DataFrame([{"disease_id": d, "disease_name": f"Disease {d[-1]}", "source": "OMIM", "hpo_id": t,
                         "negated": n, "refs": r, "evidence": "PCS", "onset": "", "frequency": "", "freq": f,
                         "sex": "", "modifier": "", "aspect": "P", "biocuration": ""} for d, t, f, n, r in rows])
    return terms, edges, ann


@pytest.fixture(scope="session")
def tiny(tiny_tables):
    terms, edges, ann = tiny_tables
    onto = Ontology.from_tables(terms, edges)
    prof = P.build(ann, onto)
    return {"onto": onto, "prof": prof, "ann": ann, "ic": P.information_content(prof), "bg": P.background(prof)}


def real_data_available() -> bool:
    need = [config.PROCESSED / "diseases.parquet", config.MODELS / "fusion_meta.json",
            config.INDEX / "passage_index.pkl", config.INDEX / "passages.parquet"]
    return all(p.exists() for p in need)


requires_data = pytest.mark.skipif(not real_data_available(), reason="run make data index train first")


@pytest.fixture(scope="session")
def copilot():
    if not real_data_available():
        pytest.skip("real data not built")
    from odyssey_anchor.agent.audit import AuditLog
    from odyssey_anchor.agent.copilot import Copilot
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp()) / "audit.jsonl"
    return Copilot(audit=AuditLog(tmp))
