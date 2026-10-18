"""Parse the Human Phenotype Ontology (obographs JSON) into flat tables.

Outputs
  hpo_terms.parquet  one row per current or obsolete term
  hpo_edges.parquet  is_a edges between current terms (child, parent)
  hpo_alt.parquet    alternative and obsolete identifiers mapped to current ones
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config

OBO = "http://purl.obolibrary.org/obo/"
ALT = "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId"
REPLACED_BY = "http://purl.obolibrary.org/obo/IAO_0100001"


def curie(iri: str) -> str:
    if iri.startswith(OBO):
        local = iri[len(OBO):]
        if "_" in local:
            prefix, num = local.split("_", 1)
            return f"{prefix}:{num}"
    return iri


def parse(path: Path | None = None) -> dict[str, pd.DataFrame]:
    path = path or config.RAW / "hp.json"
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    terms, alts = [], []
    for node in graph["nodes"]:
        tid = curie(node["id"])
        if not tid.startswith("HP:"):
            continue
        meta = node.get("meta", {})
        synonyms, syn_types = [], []
        for s in meta.get("synonyms", []):
            synonyms.append(s["val"])
            kind = s.get("synonymType", "").rsplit("#", 1)[-1] if s.get("synonymType") else ""
            syn_types.append(f"{s['pred']}|{kind}")
        replaced_by = None
        for pv in meta.get("basicPropertyValues", []):
            if pv["pred"] == ALT:
                alts.append({"alt_id": pv["val"], "id": tid})
            elif pv["pred"] == REPLACED_BY:
                replaced_by = pv["val"]
        xrefs = [x["val"] for x in meta.get("xrefs", [])]
        terms.append({
            "id": tid,
            "label": node.get("lbl", ""),
            "definition": meta.get("definition", {}).get("val", ""),
            "definition_refs": [x for x in meta.get("definition", {}).get("xrefs", []) if x.startswith("PMID")],
            "comment": " ".join(meta.get("comments", [])),
            "synonyms": synonyms,
            "synonym_types": syn_types,
            "xrefs": xrefs,
            "obsolete": bool(meta.get("deprecated", False)),
            "replaced_by": replaced_by,
        })
    terms_df = pd.DataFrame(terms)
    current = set(terms_df.loc[~terms_df.obsolete, "id"])
    edges = [
        {"child": curie(e["sub"]), "parent": curie(e["obj"])}
        for e in graph["edges"]
        if e["pred"] == "is_a"
    ]
    edges_df = pd.DataFrame(edges)
    edges_df = edges_df[edges_df.child.isin(current) & edges_df.parent.isin(current)].drop_duplicates()
    obsolete_map = terms_df.loc[terms_df.obsolete & terms_df.replaced_by.notna(), ["id", "replaced_by"]]
    alt_df = pd.concat([
        pd.DataFrame(alts),
        obsolete_map.rename(columns={"id": "alt_id", "replaced_by": "id"}),
    ]).drop_duplicates("alt_id")
    alt_df = alt_df[alt_df.id.isin(current)]
    return {"hpo_terms": terms_df, "hpo_edges": edges_df.reset_index(drop=True), "hpo_alt": alt_df.reset_index(drop=True)}


def id_mapper(terms: pd.DataFrame, alt: pd.DataFrame):
    """Return a function that maps any HPO identifier to its current identifier, or None."""
    current = set(terms.loc[~terms.obsolete, "id"])
    alt_map = dict(zip(alt.alt_id, alt.id))

    def resolve(tid: str) -> str | None:
        if tid in current:
            return tid
        return alt_map.get(tid)

    return resolve
