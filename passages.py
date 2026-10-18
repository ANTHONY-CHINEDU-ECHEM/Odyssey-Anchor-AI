"""Knowledge retrieval over the 112,000 passage knowledge base.

Candidates come from three routes: BM25 over passage text, latent semantic
similarity, and the graph route (every passage attached to a disease, gene or
phenotype that the question names). A logistic ranker fitted on development
questions orders them using eleven features:

  bm25, lsa                      lexical and semantic similarity
  ent_disease, ent_gene, ent_term the passage belongs to an entity the question names
  kind_fit                       the passage kind answers the question's intent
  system_fit                     the passage organ system holds the named phenotype
  title_overlap                  token overlap between question and passage title
  is_overview, is_system         passage subtype
  in_context                     the passage disease is in the current differential
"""
from __future__ import annotations

import json
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import config
from ..ontology.graph import Ontology
from .bm25 import BM25
from .lsa import LSA
from .text import Lexicon, tokens

INTENTS = {
    "inheritance": re.compile(r"\binherit|\bmode of transmission|\bdominant\b|\brecessive\b", re.I),
    "diseases_of_gene": re.compile(r"\b(caused by|due to|conditions?|diseases?|disorders?)\b.*\b(variants?|mutations?|in)\b\s+[A-Z0-9-]{2,}", re.I),
    "genes_of_disease": re.compile(r"\bgenes?\b", re.I),
    "feature_in_disease": re.compile(r"\b(part of|seen in|feature of|features of|occur in|common in|how often|frequency|present in|typical of)\b", re.I),
    "define_feature": re.compile(r"\b(what is meant by|meaning of|define|definition of|what does|what is)\b", re.I),
    "group": re.compile(r"\b(group|family of|types of|forms of|spectrum)\b", re.I),
}
FEATURES = ["bm25", "lsa", "ent_disease", "ent_gene", "ent_term", "kind_fit", "system_fit",
            "title_overlap", "is_overview", "is_system", "in_context"]
GENE_TOKEN = re.compile(r"\b[A-Z][A-Z0-9-]{1,9}\b")


def intent_of(question: str) -> str:
    for name in ("inheritance", "diseases_of_gene", "genes_of_disease", "feature_in_disease", "group", "define_feature"):
        if INTENTS[name].search(question):
            return name
    return "other"


def kind_fit(intent: str, pid: str) -> float:
    kind = pid.split("|", 1)[0]
    if intent in ("inheritance", "genes_of_disease"):
        return float(kind == "DIS" and pid.endswith("|overview"))
    if intent == "feature_in_disease":
        return float(kind == "DIS" and not pid.endswith("|overview"))
    if intent == "define_feature":
        return float(kind == "HPO")
    if intent == "diseases_of_gene":
        return float(kind == "GENE")
    if intent == "group":
        return float(kind == "GRP")
    return 0.0


class EntityLinker:
    def __init__(self, diseases: pd.DataFrame, genes: set[str], onto: Ontology, recognizer_lexicon: Lexicon):
        self.dlex = Lexicon(max_len=12)
        for r in diseases.itertuples():
            self.dlex.add(r.name, r.disease_id)
            for s in r.synonyms:
                if len(s) >= 3:
                    self.dlex.add(s, r.disease_id)
        self.genes = genes
        self.tlex = recognizer_lexicon
        self.onto = onto

    def link(self, question: str) -> dict[str, list[str]]:
        dis = []
        for m in self.dlex.find(question):
            dis.extend(m["concepts"])
        genes = [g for g in GENE_TOKEN.findall(question) if g in self.genes]
        terms = []
        for m in self.tlex.find(question):
            terms.extend(m["concepts"])
        return {"diseases": list(dict.fromkeys(dis)), "genes": list(dict.fromkeys(genes)),
                "terms": list(dict.fromkeys(terms))}


class PassageIndex:
    def __init__(self, passages: pd.DataFrame, linker: EntityLinker, onto: Ontology):
        self.p = passages.reset_index(drop=True)
        self.pid_row = {p: i for i, p in enumerate(self.p.pid)}
        self.linker = linker
        self.onto = onto
        self.by_disease = self.p.groupby("disease_id").apply(lambda g: g.index.to_numpy(), include_groups=False).to_dict()
        self.title_tokens = [set(tokens(t)) for t in self.p.title]
        self.system_of = {pid: pid.split("|")[2] for pid in self.p.pid if pid.startswith("DIS|") and not pid.endswith("|overview")}
        self.model: LogisticRegression | None = None

    def fit_text(self) -> "PassageIndex":
        toks = [tokens(f"{a} {b}") for a, b in zip(self.p.title, self.p.text)]
        self.bm25 = BM25().fit(toks)
        self.lsa = LSA(dims=256).fit(toks)
        return self

    # candidates and features ---------------------------------------------------
    def candidates(self, question: str, context: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        q = tokens(question)
        bm = self.bm25.score(q)
        ls = self.lsa.score(q)
        ents = self.linker.link(question)
        intent = intent_of(question)
        pick = set(np.argpartition(-bm, 100)[:100].tolist()) | set(np.argpartition(-ls, 50)[:50].tolist())
        for d in ents["diseases"]:
            pick.update(self.by_disease.get(d, []).tolist())
        for g in ents["genes"]:
            if f"GENE|{g}" in self.pid_row:
                pick.add(self.pid_row[f"GENE|{g}"])
        for t in ents["terms"]:
            if f"HPO|{t}" in self.pid_row:
                pick.add(self.pid_row[f"HPO|{t}"])
        for d in (context or []):
            if intent in ("feature_in_disease", "genes_of_disease", "inheritance"):
                pick.update(self.by_disease.get(d, []).tolist())
        idx = np.array(sorted(pick))
        sub = self.p.iloc[idx]
        dset, gset, tset = set(ents["diseases"]), set(ents["genes"]), set(ents["terms"])
        term_systems = {self.onto.systems[self.onto.index[t]] for t in tset if t in self.onto.index}
        qset = set(q)
        rows = []
        bm_max = max(float(bm.max()), 1e-6)
        ctx = set(context or [])
        for i, r in zip(idx, sub.itertuples()):
            pid = r.pid
            rows.append({
                "row": int(i), "pid": pid,
                "bm25": float(bm[i]) / bm_max, "lsa": float(ls[i]),
                "ent_disease": float(r.disease_id in dset) if r.disease_id else 0.0,
                "ent_gene": float(bool(gset) and (pid in {f"GENE|{g}" for g in gset} or bool(gset & set(r.genes)))),
                "ent_term": float(bool(tset) and (pid in {f"HPO|{t}" for t in tset} or bool(tset & set(r.hpo_ids)))),
                "kind_fit": kind_fit(intent, pid),
                "system_fit": float(self.system_of.get(pid) in term_systems) if pid in self.system_of else 0.0,
                "title_overlap": len(qset & self.title_tokens[i]) / max(len(self.title_tokens[i]), 1),
                "is_overview": float(pid.endswith("|overview")),
                "is_system": float(pid in self.system_of),
                "in_context": float(r.disease_id in ctx) if r.disease_id else 0.0,
            })
        return pd.DataFrame(rows), {"intent": intent, **ents}

    def score_frame(self, df: pd.DataFrame, method: str = "learned") -> np.ndarray:
        if method == "bm25":
            return df.bm25.to_numpy()
        if method == "lsa":
            return df.lsa.to_numpy()
        if method == "hybrid":
            return 0.5 * df.bm25.to_numpy() + 0.5 * df.lsa.to_numpy()
        X = df[FEATURES].to_numpy(np.float64)
        if method.startswith("learned_without_"):
            drop = method[len("learned_without_"):]
            X = X.copy()
            X[:, FEATURES.index(drop)] = 0.0
            return self.ablated[drop].decision_function(X)
        return self.model.decision_function(X)

    def search(self, question: str, k: int = 5, context: list[str] | None = None, method: str = "learned") -> tuple[pd.DataFrame, dict]:
        df, info = self.candidates(question, context)
        df["score"] = self.score_frame(df, method)
        df = df.sort_values("score", ascending=False).head(k).reset_index(drop=True)
        df = df.merge(self.p[["pid", "kind", "title", "text", "disease_id", "refs"]], on="pid", how="left")
        return df, info

    # learning ---------------------------------------------------------------------
    def fit_ranker(self, questions: pd.DataFrame) -> dict:
        frames = []
        for q in contextual_variants(questions, sorted(self.by_disease)):
            df, _ = self.candidates(q["question"], q["context"])
            df["label"] = df.pid.isin(set(q["gold"])).astype(int)
            df["qid"] = q["qid"]
            frames.append(df)
        D = pd.concat(frames, ignore_index=True)
        self.model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(D[FEATURES].to_numpy(np.float64), D.label.to_numpy())
        self.ablated = {}
        for f in ("kind_fit", "ent_disease", "lsa"):
            X = D[FEATURES].to_numpy(np.float64).copy()
            X[:, FEATURES.index(f)] = 0.0
            self.ablated[f] = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, D.label.to_numpy())
        recall = D.groupby("qid").label.max().mean()
        return {"weights": dict(zip(FEATURES, self.model.coef_.ravel().round(4).tolist())),
                "candidate_recall": float(recall), "rows": int(len(D))}

    def save(self) -> None:
        with open(config.INDEX / "passage_index.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "lsa": self.lsa, "model": self.model, "ablated": self.ablated}, f)

    def load_models(self) -> "PassageIndex":
        with open(config.INDEX / "passage_index.pkl", "rb") as f:
            obj = pickle.load(f)
        self.bm25, self.lsa, self.model, self.ablated = obj["bm25"], obj["lsa"], obj["model"], obj["ablated"]
        return self


def contextual_variants(questions: pd.DataFrame, disease_pool: list[str], seed: int = config.SEED) -> list[dict]:
    """Each question as asked, plus an in session version that names "the leading candidate"
    instead of the disease and passes the leading disease as context, as the copilot does."""
    out = []
    for q in questions.itertuples():
        out.append({"qid": q.qid, "question": q.question, "gold": list(q.gold), "context": None, "kind": q.kind})
        if not isinstance(q.disease_id, str) or q.kind not in ("genes_of_disease", "inheritance", "feature_in_disease"):
            continue
        if q.kind == "genes_of_disease":
            text = "which genes are linked to the leading candidate?"
        elif q.kind == "inheritance":
            text = "how is the leading candidate inherited"
        else:
            text = q.question.split(" part of ")[0] + " part of the leading candidate"
        # The copilot passes only the leading concept's records when a question names "the leading candidate".
        out.append({"qid": q.qid + "c", "question": text, "gold": list(q.gold), "context": [q.disease_id], "kind": q.kind + "_in_session"})
    return out


def build(onto: Ontology, recognizer_lexicon: Lexicon) -> tuple[PassageIndex, dict]:
    passages = pd.read_parquet(config.INDEX / "passages.parquet")
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet")
    genes = set(pd.read_parquet(config.PROCESSED / "disease_genes.parquet").gene_symbol)
    linker = EntityLinker(diseases, genes, onto, recognizer_lexicon)
    idx = PassageIndex(passages, linker, onto).fit_text()
    q = pd.read_json(config.EVAL / "questions.jsonl", lines=True)
    info = idx.fit_ranker(q[q.split.eq("dev")])
    idx.save()
    (config.MODELS / "passage_ranker.json").write_text(json.dumps(info, indent=2))
    return idx, info


def load(onto: Ontology, recognizer_lexicon: Lexicon) -> PassageIndex:
    passages = pd.read_parquet(config.INDEX / "passages.parquet")
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet")
    genes = set(pd.read_parquet(config.PROCESSED / "disease_genes.parquet").gene_symbol)
    linker = EntityLinker(diseases, genes, onto, recognizer_lexicon)
    return PassageIndex(passages, linker, onto).load_models()
