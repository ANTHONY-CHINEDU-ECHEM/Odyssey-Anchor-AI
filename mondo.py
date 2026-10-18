"""Parse the Mondo disease ontology.

Mondo supplies the unified disease identity: names, synonyms, definitions,
cross references to OMIM and Orphanet, and the disease hierarchy used for
grouping candidates into families. Sentences about treatment or management are
removed from definitions at ingestion: Odyssey Anchor AI supports diagnosis,
never treatment, so treatment text never enters the knowledge base.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .. import config
from .hpo import curie

TREATMENT = re.compile(
    r"\b(treat|treated|treatment|treatments|therapy|therapies|therapeutic|management|managed|"
    r"dose|doses|dosage|drug|drugs|medication|medications|surgery|surgical|transplant\w*|"
    r"prognosis|cure|cured|supportive care)\b",
    re.IGNORECASE,
)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def strip_treatment(text: str) -> tuple[str, int]:
    """Drop every sentence that talks about treatment. Returns the text and the number removed."""
    if not text:
        return "", 0
    kept, removed = [], 0
    for sentence in SENTENCE.split(text.strip()):
        if TREATMENT.search(sentence):
            removed += 1
        else:
            kept.append(sentence)
    return " ".join(kept), removed


def normalise_xref(x: str) -> str | None:
    if x.startswith("OMIM:") and not x.startswith("OMIM:PS"):
        return x
    if x.startswith("Orphanet:"):
        return "ORPHA:" + x.split(":", 1)[1]
    if x.startswith("DECIPHER:"):
        return x
    return None


def parse(path: Path | None = None) -> dict[str, pd.DataFrame]:
    path = path or config.RAW / "mondo.json"
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    classes, xrefs = [], []
    removed_total = 0
    for node in graph["nodes"]:
        mid = curie(node["id"])
        if not mid.startswith("MONDO:"):
            continue
        meta = node.get("meta", {})
        if meta.get("deprecated"):
            continue
        definition, removed = strip_treatment(meta.get("definition", {}).get("val", ""))
        removed_total += removed
        subsets = [s.rsplit("#", 1)[-1] for s in meta.get("subsets", [])]
        classes.append({
            "mondo_id": mid,
            "label": node.get("lbl", ""),
            "definition": definition,
            "definition_source": ";".join(meta.get("definition", {}).get("xrefs", [])[:3]),
            "synonyms": sorted({s["val"] for s in meta.get("synonyms", []) if s.get("pred") == "hasExactSynonym"}),
            "rare": "rare" in subsets,
        })
        for x in meta.get("xrefs", []):
            target = normalise_xref(x["val"])
            if target:
                xrefs.append({"mondo_id": mid, "xref": target})
    classes_df = pd.DataFrame(classes)
    valid = set(classes_df.mondo_id)
    edges = [
        {"child": curie(e["sub"]), "parent": curie(e["obj"])}
        for e in graph["edges"]
        if e["pred"] == "is_a"
    ]
    edges_df = pd.DataFrame(edges)
    edges_df = edges_df[edges_df.child.isin(valid) & edges_df.parent.isin(valid)].drop_duplicates()
    classes_df.attrs["treatment_sentences_removed"] = removed_total
    return {
        "mondo_classes": classes_df,
        "mondo_edges": edges_df.reset_index(drop=True),
        "mondo_xrefs": pd.DataFrame(xrefs).drop_duplicates().reset_index(drop=True),
    }
