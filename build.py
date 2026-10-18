"""Build the knowledge base: passages for retrieval and one document per disease.

Passage kinds
  DIS overview   identity, synonyms, Mondo definition, inheritance, onset, genes
  DIS system     one per disease and organ system: every annotated feature with its
                 frequency and the references it rests on, and features documented absent
  HPO            one per phenotype term: definition, synonyms, parents, organ system
  GENE           one per gene: the diseases it is linked to and their inheritance
  GRP            one per Mondo disease group with a definition that covers two or
                 more annotated diseases

Every passage carries the identifiers it mentions (diseases, terms, genes) and
the references behind it, which the verifier uses to check citations.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from .. import config
from ..retrieval.text import tokens
from .ids import disease_overview_id, disease_system_id, gene_id, group_id, term_id

FREQ_WORDS = {
    "HP:0040280": "obligate (100%)",
    "HP:0040281": "very frequent (80 to 99%)",
    "HP:0040282": "frequent (30 to 79%)",
    "HP:0040283": "occasional (5 to 29%)",
    "HP:0040284": "very rare (1 to 4%)",
}


def frequency_phrase(raw: str) -> str:
    if not raw:
        return "frequency not stated"
    if raw in FREQ_WORDS:
        return FREQ_WORDS[raw]
    if "/" in raw:
        n, m = raw.split("/", 1)
        return f"{n} of {m} reported patients"
    if raw.endswith("%"):
        return f"{raw} of reported patients"
    return raw


def _join(items: list[str], limit: int | None = None) -> str:
    items = list(items)
    if limit is not None and len(items) > limit:
        items = items[:limit] + [f"and {len(items) - limit} more"]
    return ", ".join(items)


def build_passages(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    terms = t["hpo_terms"]
    terms = terms[~terms.obsolete]
    label = dict(zip(terms.id, terms.label))
    system = dict(zip(terms.id, terms.system))
    diseases = t["diseases"].copy()
    for col in ("mondo_id", "mondo_label", "definition", "definition_source"):
        diseases[col] = diseases[col].fillna("")
    ann = t["annotations"]
    pheno = ann[ann.aspect.eq("P")]
    out = []

    # Disease overviews
    for r in diseases.itertuples():
        bits = [f"{r.name} ({r.disease_id}{'; ' + r.mondo_id if r.mondo_id else ''})."]
        if len(r.synonyms):
            bits.append(f"Also known as {_join(r.synonyms, 8)}.")
        if r.definition:
            bits.append(r.definition)
        if len(r.inheritance):
            bits.append(f"Mode of inheritance: {_join([x.lower() for x in r.inheritance])}.")
        if len(r.onset):
            bits.append(f"Onset: {_join([x.lower() for x in r.onset])}.")
        if len(r.genes):
            bits.append(f"Associated genes: {_join(r.genes)}.")
        bits.append(f"The HPO annotates {r.n_phenotypes} phenotypic features for this disease.")
        out.append({
            "pid": disease_overview_id(r.disease_id), "kind": "DIS", "title": f"{r.name}: overview",
            "text": " ".join(bits), "disease_id": r.disease_id, "hpo_ids": [],
            "genes": list(r.genes), "refs": [r.disease_id] + ([r.definition_source] if r.definition_source else []),
        })

    # Disease by organ system
    pheno = pheno.assign(system=pheno.hpo_id.map(system).fillna(""))
    pheno = pheno[pheno.system.ne("")]
    names = dict(zip(diseases.disease_id, diseases.name))
    for (did, sys), g in pheno.groupby(["disease_id", "system"], sort=True):
        pos = g[~g.negated].sort_values(["freq", "hpo_id"], ascending=[False, True])
        neg = g[g.negated]
        lines, refs, hpo_ids = [], set(), []
        for row in pos.itertuples():
            lines.append(f"{label.get(row.hpo_id, row.hpo_id)} ({frequency_phrase(row.frequency)})")
            refs.update(row.refs)
            hpo_ids.append(row.hpo_id)
        text = f"{names.get(did, did)}, {label.get(sys, sys).lower()}: "
        text += (_join(lines) + ".") if lines else "no positive features annotated."
        if len(neg):
            text += " Documented as absent: " + _join([label.get(x, x) for x in neg.hpo_id]) + "."
            hpo_ids += neg.hpo_id.tolist()
            for rr in neg.refs:
                refs.update(rr)
        out.append({
            "pid": disease_system_id(did, sys), "kind": "DIS", "title": f"{names.get(did, did)}: {label.get(sys, sys)}",
            "text": text, "disease_id": did, "hpo_ids": hpo_ids, "genes": [], "refs": sorted(refs)[:40],
        })

    # Phenotype terms
    parents = t["hpo_edges"].groupby("child").parent.apply(list).to_dict()
    n_dis = pheno[~pheno.negated].groupby("hpo_id").disease_id.nunique()
    for r in terms.itertuples():
        bits = [f"{r.label} ({r.id})."]
        if r.definition:
            bits.append(r.definition)
        syns = [s for s in r.synonyms if s.lower() != r.label.lower()]
        if syns:
            bits.append(f"Synonyms: {_join(syns, 10)}.")
        ps = [label.get(p, p) for p in parents.get(r.id, [])]
        if ps:
            bits.append(f"A kind of {_join(ps)}.")
        if r.system and r.system != r.id:
            bits.append(f"Organ system: {label.get(r.system, r.system).lower()}.")
        if r.id in n_dis.index:
            bits.append(f"Directly annotated to {int(n_dis[r.id])} diseases.")
        out.append({
            "pid": term_id(r.id), "kind": "HPO", "title": r.label, "text": " ".join(bits),
            "disease_id": None, "hpo_ids": [r.id], "genes": [], "refs": list(r.definition_refs),
        })

    # Genes
    dg = t["disease_genes"].merge(diseases[["disease_id", "name", "inheritance"]], on="disease_id", how="inner")
    for gene, g in dg.groupby("gene_symbol"):
        items = []
        for r in g.itertuples():
            inh = f", {r.inheritance[0].lower()}" if len(r.inheritance) else ""
            items.append(f"{r.name} ({r.disease_id}{inh})")
        out.append({
            "pid": gene_id(gene), "kind": "GENE", "title": f"{gene}",
            "text": f"{gene} is linked to {len(items)} disease{'s' if len(items) != 1 else ''}: {_join(items, 25)}.",
            "disease_id": None, "hpo_ids": [], "genes": [gene], "refs": g.disease_id.tolist()[:25],
        })

    # Mondo groups
    classes = t["mondo_classes"].set_index("mondo_id")
    edges = t["mondo_edges"]
    up = edges.groupby("child").parent.apply(list).to_dict()
    members: dict[str, set[str]] = defaultdict(set)
    for r in diseases.dropna(subset=["mondo_id"]).itertuples():
        stack, seen = [r.mondo_id], set()
        while stack:
            m = stack.pop()
            for p in up.get(m, []):
                if p not in seen:
                    seen.add(p)
                    members[p].add(r.disease_id)
                    stack.append(p)
    for mid, ds in members.items():
        if not (2 <= len(ds) <= 400) or mid not in classes.index:
            continue
        c = classes.loc[mid]
        if not c.definition:
            continue
        listed = sorted(ds, key=lambda d: names.get(d, d))
        text = f"{c.label} ({mid}) is a disease group. {c.definition} It includes {len(ds)} annotated diseases: "
        text += _join([f"{names.get(d, d)} ({d})" for d in listed], 20) + "."
        out.append({
            "pid": group_id(mid), "kind": "GRP", "title": c.label, "text": text,
            "disease_id": None, "hpo_ids": [], "genes": [], "refs": listed[:20],
        })
    df = pd.DataFrame(out)
    df["tokens"] = (df.title + " " + df.text).map(tokens)
    return df


def disease_document(name: str, synonyms, definition: str, feature_labels: list[str]) -> list[str]:
    """Token document for one disease, used by the text based diagnostic rankers."""
    text = " ".join([name, name, " ".join(synonyms), definition, " . ".join(feature_labels)])
    return tokens(text)


def build_disease_docs(t: dict[str, pd.DataFrame], annotations: pd.DataFrame | None = None,
                       disease_ids: list[str] | None = None) -> dict[str, list[str]]:
    terms = t["hpo_terms"]
    label = dict(zip(terms.id, terms.label))
    ann = annotations if annotations is not None else t["annotations"]
    pheno = ann[ann.aspect.eq("P") & ~ann.negated]
    feats = pheno.groupby("disease_id").hpo_id.apply(lambda s: [label.get(x, x) for x in dict.fromkeys(s)]).to_dict()
    d = t["diseases"].fillna({"definition": ""}).set_index("disease_id")
    ids = disease_ids if disease_ids is not None else d.index.tolist()
    return {i: disease_document(d.at[i, "name"], d.at[i, "synonyms"], d.at[i, "definition"], feats.get(i, []))
            for i in ids}


def load_tables(names=("hpo_terms", "hpo_edges", "annotations", "diseases", "disease_genes",
                       "mondo_classes", "mondo_edges")) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(config.PROCESSED / f"{n}.parquet") for n in names}


def run() -> dict:
    t = load_tables()
    passages = build_passages(t)
    passages.drop(columns=["tokens"]).to_parquet(config.INDEX / "passages.parquet", index=False)
    stats = passages.kind.value_counts().to_dict()
    stats["total"] = int(len(passages))
    stats["tokens"] = int(passages.tokens.map(len).sum())
    (config.INDEX / "passages_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    config.ensure_dirs()
    print(json.dumps(run(), indent=2))
