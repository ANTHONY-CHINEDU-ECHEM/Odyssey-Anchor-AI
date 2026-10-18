"""Parse the GA4GH phenopacket store into one row per published patient.

Each phenopacket is a real individual described in a peer reviewed paper, with
observed and explicitly excluded HPO features, a confirmed diagnosis and the
causative gene. These are the evaluation cases. They are never used to build
the knowledge base.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from .. import config

DURATION = re.compile(r"P(?:(?P<y>\d+)Y)?(?:(?P<m>\d+)M)?(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?")


def age_years(time_element: dict | None) -> float | None:
    """Convert a phenopacket TimeElement to years. Prenatal ages map to zero."""
    if not time_element:
        return None
    if "gestationalAge" in time_element:
        return 0.0
    iso = time_element.get("age", {}).get("iso8601duration")
    if not iso:
        return None
    m = DURATION.fullmatch(iso)
    if not m:
        return None
    y = float(m.group("y") or 0)
    mo = float(m.group("m") or 0)
    w = float(m.group("w") or 0)
    d = float(m.group("d") or 0)
    return round(y + mo / 12 + w / 52.18 + d / 365.25, 3)


def parse_one(doc: dict, cohort: str, resolve) -> dict | None:
    observed, excluded, unmapped = [], [], 0
    for feature in doc.get("phenotypicFeatures", []):
        tid = resolve(feature["type"]["id"]) if resolve else feature["type"]["id"]
        if tid is None:
            unmapped += 1
            continue
        (excluded if feature.get("excluded") else observed).append(tid)
    diagnoses, genes, variants = [], set(), []
    for interp in doc.get("interpretations", []):
        dx = interp.get("diagnosis", {})
        if "disease" in dx:
            diagnoses.append((dx["disease"]["id"], dx["disease"].get("label", "")))
        for gi in dx.get("genomicInterpretations", []):
            vd = gi.get("variantInterpretation", {}).get("variationDescriptor", {})
            sym = vd.get("geneContext", {}).get("symbol")
            if sym:
                genes.add(sym)
            for ex in vd.get("expressions", []):
                if ex.get("syntax") == "hgvs.c":
                    variants.append(ex["value"])
    if not diagnoses:
        return None
    refs = doc.get("metaData", {}).get("externalReferences", [])
    pmid = next((r["id"] for r in refs if r["id"].startswith("PMID:")), None)
    title = next((r.get("description", "") for r in refs if r["id"].startswith("PMID:")), "")
    subject = doc.get("subject", {})
    observed = list(dict.fromkeys(observed))
    excluded = [t for t in dict.fromkeys(excluded) if t not in observed]
    return {
        "case_id": doc["id"],
        "cohort": cohort,
        "pmid": pmid,
        "paper_title": title,
        "sex": subject.get("sex", "UNKNOWN_SEX"),
        "age_years": age_years(subject.get("timeAtLastEncounter")),
        "observed": observed,
        "excluded": excluded,
        "n_observed": len(observed),
        "n_excluded": len(excluded),
        "unmapped_terms": unmapped,
        "disease_id": diagnoses[0][0],
        "disease_label": diagnoses[0][1],
        "genes": sorted(genes),
        "variants": variants[:4],
    }


def parse(path: Path | None = None, resolve=None) -> pd.DataFrame:
    path = path or config.RAW / "all_phenopackets.zip"
    rows = []
    with zipfile.ZipFile(path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json"))
        release = names[0].split("/")[0] if names else None
        for name in names:
            parts = name.split("/")
            cohort = parts[1] if len(parts) > 2 else "unknown"
            row = parse_one(json.loads(z.read(name)), cohort, resolve)
            if row:
                row["source_file"] = name
                rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates("case_id").reset_index(drop=True)
    df.attrs["release"] = release
    return df
