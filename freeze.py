"""Freeze and hash the evaluation sets before any index or model exists.

Cases
  Eligible cases have a diagnosis present in the knowledge base and at least
  three observed features. Cohorts (one folder per gene in the phenopacket
  store) are joined into groups whenever they share a disease, and a quarter of
  the groups are held out as the blind set. Blind diseases and genes are never
  seen while any model or setting is chosen.

Questions
  Knowledge questions are generated from the processed tables with stable gold
  passage identifiers, phrased through synonyms rather than canonical labels so
  retrieval cannot rely on exact title matches.

The blind set is hashed. evaluate_blind refuses to run if the hash changes,
and every blind run is appended to blind_runs.json.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..knowledge.ids import (
    disease_overview_id, disease_system_id, gene_id, term_id,
)


def _groups(cases: pd.DataFrame) -> dict[str, int]:
    """Union cohorts that share a disease so no disease spans both splits."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _, grp in cases.groupby("disease_id"):
        cohorts = sorted(grp.cohort.unique())
        for c in cohorts[1:]:
            union(cohorts[0], c)
    roots = sorted({find(c) for c in cases.cohort.unique()})
    index = {r: i for i, r in enumerate(roots)}
    return {c: index[find(c)] for c in cases.cohort.unique()}


def split_cases(cases: pd.DataFrame, diseases: pd.DataFrame) -> pd.DataFrame:
    known = set(diseases.disease_id)
    eligible = cases[cases.disease_id.isin(known) & (cases.n_observed >= config.MIN_FEATURES)].copy()
    group_of = _groups(eligible)
    eligible["group"] = eligible.cohort.map(group_of)
    groups = sorted(eligible.group.unique())
    rng = random.Random(config.SEED)
    rng.shuffle(groups)
    n_blind = int(round(len(groups) * config.BLIND_FRACTION))
    blind_groups = set(groups[:n_blind])
    eligible["split"] = np.where(eligible.group.isin(blind_groups), "blind", "dev")
    # Development folds for grouped cross validation, by group.
    dev_groups = [g for g in groups if g not in blind_groups]
    fold_of = {g: i % 5 for i, g in enumerate(dev_groups)}
    eligible["fold"] = eligible.group.map(fold_of).fillna(-1).astype(int)
    return eligible.reset_index(drop=True)


def make_questions(tables: dict[str, pd.DataFrame], split: pd.DataFrame, n_per_kind: int = 110) -> pd.DataFrame:
    """Template knowledge questions with stable gold passage identifiers."""
    rng = random.Random(config.SEED + 1)
    terms = tables["hpo_terms"]
    terms = terms[~terms.obsolete]
    diseases = tables["diseases"]
    ann = tables["annotations"]
    ann = ann[ann.aspect.eq("P") & ~ann.negated]
    labels = dict(zip(terms.id, terms.label))
    system_of = dict(zip(terms.id, terms.system))
    blind_dx = set(split.loc[split.split.eq("blind"), "disease_id"])
    dev_dx = set(split.loc[split.split.eq("dev"), "disease_id"])

    def synonym_or_label(syns, label):
        exact = [s for s in syns if s.lower() != label.lower() and 3 < len(s) < 60]
        return rng.choice(exact) if exact else label

    def which_split(did: str) -> str:
        if did in blind_dx:
            return "blind"
        if did in dev_dx:
            return "dev"
        return "blind" if rng.random() < 0.5 else "dev"

    q = []
    with_genes = diseases[diseases.genes.str.len().gt(0) & diseases.synonyms.str.len().gt(0)]
    for _, r in with_genes.sample(n=min(n_per_kind, len(with_genes)), random_state=config.SEED).iterrows():
        name = synonym_or_label(list(r.synonyms), r["name"])
        q.append(("genes_of_disease", f"which genes are linked to {name}?", [disease_overview_id(r.disease_id)], r.disease_id))
    with_inh = diseases[diseases.inheritance.str.len().gt(0) & diseases.synonyms.str.len().gt(0)]
    for _, r in with_inh.sample(n=min(n_per_kind, len(with_inh)), random_state=config.SEED + 2).iterrows():
        name = synonym_or_label(list(r.synonyms), r["name"])
        q.append(("inheritance", f"how is {name} inherited", [disease_overview_id(r.disease_id)], r.disease_id))
    with_syn = terms[terms.synonyms.str.len().gt(0) & terms.definition.str.len().gt(20)]
    for _, r in with_syn.sample(n=n_per_kind, random_state=config.SEED + 3).iterrows():
        name = synonym_or_label(list(r.synonyms), r.label)
        q.append(("define_feature", f"what is meant by {name.lower()}", [term_id(r.id)], None))
    # Feature in disease: pick a specific, frequent feature of a disease.
    sample = ann[ann.freq.ge(0.5)].sample(n=n_per_kind * 4, random_state=config.SEED + 4)
    seen = set()
    for _, r in sample.iterrows():
        if len([x for x in q if x[0] == "feature_in_disease"]) >= n_per_kind:
            break
        sys = system_of.get(r.hpo_id, "")
        if not sys or (r.disease_id, sys) in seen:
            continue
        seen.add((r.disease_id, sys))
        drow = diseases[diseases.disease_id.eq(r.disease_id)]
        if drow.empty:
            continue
        dname = synonym_or_label(list(drow.iloc[0].synonyms), drow.iloc[0]["name"])
        q.append(("feature_in_disease", f"is {labels.get(r.hpo_id, r.hpo_id).lower()} part of {dname}",
                  [disease_system_id(r.disease_id, sys)], r.disease_id))
    genes = tables["disease_genes"].gene_symbol.value_counts()
    multi = genes[genes.ge(2)].index.tolist()
    rng.shuffle(multi)
    for g in multi[:n_per_kind]:
        q.append(("diseases_of_gene", f"what conditions are caused by variants in {g}", [gene_id(g)], None))

    df = pd.DataFrame(q, columns=["kind", "question", "gold", "disease_id"])
    df["split"] = [which_split(d) if d else ("blind" if rng.random() < 0.5 else "dev") for d in df.disease_id]
    df.insert(0, "qid", [f"Q{i:04d}" for i in range(len(df))])
    return df


def _hash_frame(df: pd.DataFrame, cols: list[str]) -> str:
    payload = df[cols].to_json(orient="records", force_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def run() -> dict:
    config.ensure_dirs()
    tables = {n: pd.read_parquet(config.PROCESSED / f"{n}.parquet") for n in
              ("cases", "diseases", "hpo_terms", "annotations", "disease_genes")}
    split = split_cases(tables["cases"], tables["diseases"])
    questions = make_questions(tables, split)
    split.to_parquet(config.EVAL / "case_split.parquet", index=False)
    questions.to_json(config.EVAL / "questions.jsonl", orient="records", lines=True)
    blind = split[split.split.eq("blind")].sort_values("case_id")
    blind_q = questions[questions.split.eq("blind")].sort_values("qid")
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "cohort groups joined by shared disease; 25% of groups blind; strict source paper removal at scoring",
        "seed": config.SEED,
        "cases_total": int(len(tables["cases"])),
        "cases_eligible": int(len(split)),
        "dev_cases": int(split.split.eq("dev").sum()),
        "blind_cases": int(len(blind)),
        "dev_groups": int(split.loc[split.split.eq("dev"), "group"].nunique()),
        "blind_groups": int(blind.group.nunique()),
        "blind_diseases": int(blind.disease_id.nunique()),
        "dev_diseases": int(split.loc[split.split.eq("dev"), "disease_id"].nunique()),
        "questions_dev": int(questions.split.eq("dev").sum()),
        "questions_blind": int(len(blind_q)),
        "blind_case_sha256": _hash_frame(blind.assign(
            observed=blind.observed.apply(list), excluded=blind.excluded.apply(list)),
            ["case_id", "disease_id", "observed", "excluded"]),
        "blind_question_sha256": _hash_frame(blind_q, ["qid", "question", "gold"]),
    }
    (config.EVAL / "FROZEN.json").write_text(json.dumps(frozen, indent=2))
    runs = config.EVAL / "blind_runs.json"
    if not runs.exists():
        runs.write_text(json.dumps({"runs": []}, indent=2))
    return frozen


def verify() -> dict:
    """Recompute the blind hashes and compare to the frozen record."""
    frozen = json.loads((config.EVAL / "FROZEN.json").read_text())
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    questions = pd.read_json(config.EVAL / "questions.jsonl", lines=True)
    blind = split[split.split.eq("blind")].sort_values("case_id")
    blind = blind.assign(observed=blind.observed.apply(list), excluded=blind.excluded.apply(list))
    blind_q = questions[questions.split.eq("blind")].sort_values("qid")
    case_hash = _hash_frame(blind, ["case_id", "disease_id", "observed", "excluded"])
    q_hash = _hash_frame(blind_q, ["qid", "question", "gold"])
    if case_hash != frozen["blind_case_sha256"] or q_hash != frozen["blind_question_sha256"]:
        raise RuntimeError("The blind set has changed since it was frozen")
    return frozen


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
