"""Parse the HPO disease annotation file (phenotype.hpoa).

Each row links a disease (OMIM, ORPHA or DECIPHER) to an HPO term with a
frequency, an onset, a sex restriction, a NOT qualifier and its references.
Frequencies arrive in three forms: an HPO frequency term, a count such as
7/12, or a percentage. All are converted to a probability with a light
Bayesian smoothing for small counts, so that 1/1 is not treated as certainty.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

FREQUENCY_TERMS = {
    "HP:0040280": 1.0,     # obligate, 100%
    "HP:0040281": 0.895,   # very frequent, 80 to 99%
    "HP:0040282": 0.545,   # frequent, 30 to 79%
    "HP:0040283": 0.17,    # occasional, 5 to 29%
    "HP:0040284": 0.025,   # very rare, 1 to 4%
    "HP:0040285": 0.0,     # excluded, 0%
}
DEFAULT_FREQUENCY = 0.5    # no frequency given: an uninformative middle value


def parse_frequency(raw: object) -> float:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
        return DEFAULT_FREQUENCY
    s = str(raw).strip()
    if s in FREQUENCY_TERMS:
        return FREQUENCY_TERMS[s]
    if "/" in s:
        n, m = s.split("/", 1)
        try:
            n, m = float(n), float(m)
        except ValueError:
            return DEFAULT_FREQUENCY
        if m <= 0:
            return DEFAULT_FREQUENCY
        return float((n + 0.5) / (m + 1.0))
    if s.endswith("%"):
        try:
            return min(max(float(s[:-1]) / 100.0, 0.0), 1.0)
        except ValueError:
            return DEFAULT_FREQUENCY
    return DEFAULT_FREQUENCY


def parse(path: Path | None = None, resolve=None) -> pd.DataFrame:
    path = path or config.RAW / "phenotype.hpoa"
    version = None
    skip = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                skip += 1
                if line.startswith("#version"):
                    version = line.split(":", 1)[1].strip()
            else:
                break
    df = pd.read_csv(path, sep="\t", skiprows=skip, dtype=str, keep_default_na=False)
    df = df.rename(columns={"database_id": "disease_id", "hpo_id": "hpo_id_raw"})
    if resolve is not None:
        df["hpo_id"] = df.hpo_id_raw.map(resolve)
        df = df[df.hpo_id.notna()].copy()
    else:
        df["hpo_id"] = df.hpo_id_raw
    df["negated"] = df.qualifier.eq("NOT")
    df["refs"] = df.reference.str.split(";").apply(lambda r: [x.strip() for x in r if x.strip()])
    df["freq"] = df.frequency.map(parse_frequency).astype(float)
    df.loc[df.frequency.eq("HP:0040285"), "negated"] = True
    df["source"] = df.disease_id.str.split(":").str[0]
    df.attrs["version"] = version
    cols = ["disease_id", "disease_name", "source", "hpo_id", "negated", "refs", "evidence",
            "onset", "frequency", "freq", "sex", "modifier", "aspect", "biocuration"]
    return df[cols].reset_index(drop=True)
