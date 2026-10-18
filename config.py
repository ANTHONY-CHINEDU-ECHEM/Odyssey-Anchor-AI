"""Central configuration: paths, source URLs, seeds and fixed thresholds.

Every path in the project is derived from ROOT so the pipeline can run from a
clean checkout with `make all`. Nothing here depends on the environment except
ODYSSEY_ROOT, which lets the tests point the pipeline at a temporary copy.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ODYSSEY_ROOT", Path(__file__).resolve().parents[1]))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVAL = DATA / "eval"
MODELS = DATA / "models"
INDEX = DATA / "index"
AUDIT = DATA / "audit"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
WEB = ROOT / "web"

SEED = 20261017

# Public sources. Each is a GitHub release asset or repository, which keeps the
# whole pipeline reproducible from one network allowlist.
HPO_RELEASE = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download"
MONDO_RELEASE = "https://github.com/monarch-initiative/mondo/releases/latest/download"
PHENOPACKET_RELEASE = "https://github.com/monarch-initiative/phenopacket-store/releases/latest/download"

SOURCES = {
    "hp.json": f"{HPO_RELEASE}/hp.json",
    "phenotype.hpoa": f"{HPO_RELEASE}/phenotype.hpoa",
    "genes_to_phenotype.txt": f"{HPO_RELEASE}/genes_to_phenotype.txt",
    "mondo.json": f"{MONDO_RELEASE}/mondo.json",
    "all_phenopackets.zip": f"{PHENOPACKET_RELEASE}/all_phenopackets.zip",
}

# Root of the phenotypic abnormality branch. Terms outside it (inheritance,
# onset, frequency, clinical modifiers) are metadata, not patient features.
PHENOTYPIC_ABNORMALITY = "HP:0000118"
INHERITANCE_ROOT = "HP:0000005"
ONSET_ROOT = "HP:0003674"
FREQUENCY_ROOT = "HP:0040279"

# Evaluation protocol
BLIND_FRACTION = 0.25          # share of cohorts (gene folders) held out entirely
MIN_FEATURES = 3               # cases with fewer observed features are excluded
TOP_K = (1, 3, 10, 50)
CANDIDATES = 200               # shortlist size passed to the fused ranker

# Copilot
MAX_DIFFERENTIAL = 10
MIN_LEXICAL_SUPPORT = 0.4      # verifier: share of content words found in cited passages
EXCLUDED_PENALTY_FLOOR = 0.05  # likelihood ratio floor when a disease feature is excluded


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, EVAL, MODELS, INDEX, AUDIT, REPORTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
