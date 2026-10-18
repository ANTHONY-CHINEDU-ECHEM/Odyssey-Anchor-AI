"""The diagnostic engine: every scorer over every disease, with optional strict removal.

`score` returns, for one patient, an array per signal over all diseases:
likelihood ratio terms, semantic similarity in both directions, exact overlap,
BM25 and LSA over disease documents, and prior information (profile size, sex
and onset consistency). In strict mode the annotations that rest only on the
patient's own paper are removed and the affected diseases are rescored.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..knowledge.build import build_disease_docs, load_tables
from ..ontology import profiles as P
from ..ontology.graph import Ontology
from ..retrieval.bm25 import BM25
from ..retrieval.lsa import LSA
from ..retrieval.text import tokens
from .scorers import LRParams, Patient, Scorer

ADULT_ONSET = {"Adult onset", "Late onset", "Middle age onset", "Young adult onset"}
X_RECESSIVE = "X-linked recessive inheritance"


@dataclass
class Signals:
    arrays: dict[str, np.ndarray]
    patient: Patient
    strict_diseases: list[str]
    strict_profile: P.Profiles | None = None

    def __getitem__(self, k: str) -> np.ndarray:
        return self.arrays[k]


class Engine:
    def __init__(self, tables: dict[str, pd.DataFrame] | None = None, params: LRParams | None = None):
        self.t = tables or load_tables()
        self.onto = Ontology.from_tables(self.t["hpo_terms"], self.t["hpo_edges"])
        _ = self.onto.ANC_csc
        self.prof = P.build(self.t["annotations"], self.onto)
        self.ic = P.information_content(self.prof)
        self.bg = P.background(self.prof)
        self.scorer = Scorer(self.onto, self.ic, self.bg, params)
        self.strict = P.StrictIndex(self.t["annotations"])
        self.row = {d: i for i, d in enumerate(self.prof.disease_ids)}
        d = self.t["diseases"].set_index("disease_id").reindex(self.prof.disease_ids)
        self.names = d["name"].fillna("").tolist()
        self.size = np.log1p(self.prof.size.astype(np.float32))
        self.xr = np.array([X_RECESSIVE in list(x) if isinstance(x, (list, np.ndarray)) else False
                            for x in d["inheritance"]])
        self.adult_only = np.array([
            (len(o) > 0 and set(o) <= ADULT_ONSET) if isinstance(o, (list, np.ndarray)) else False
            for o in d["onset"]
        ])
        self._text_index()

    # text indexes over disease documents --------------------------------------
    def _text_index(self) -> None:
        cache = config.INDEX / "disease_text.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                obj = pickle.load(f)
            if obj["ids"] == self.prof.disease_ids:
                self.bm25, self.lsa = obj["bm25"], obj["lsa"]
                return
        docs = build_disease_docs(self.t, disease_ids=self.prof.disease_ids)
        corpus = [docs[d] for d in self.prof.disease_ids]
        self.bm25 = BM25().fit(corpus)
        self.lsa = LSA(dims=192).fit(corpus)
        config.INDEX.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump({"ids": self.prof.disease_ids, "bm25": self.bm25, "lsa": self.lsa}, f)

    def query_tokens(self, pat: Patient) -> list[str]:
        return tokens(" . ".join(self.onto.labels[i] for i in pat.observed))

    # scoring ------------------------------------------------------------------
    def patient(self, observed, excluded=(), sex="UNKNOWN_SEX", age_years=None) -> Patient:
        return Patient.from_terms(self.onto, observed, excluded, sex, age_years)

    def _score_block(self, pat: Patient, prof: P.Profiles) -> dict[str, np.ndarray]:
        r = self.scorer.resnik(pat, prof)
        lr = self.scorer.likelihood_ratio(pat, prof)
        n_obs = max(len(pat.observed), 1)
        return {
            "lr": lr["lr"], "lr_obs": lr["lr_obs"], "lr_exc": lr["lr_exc"],
            "support": lr["support"] / n_obs, "against": lr["against"] / n_obs,
            "not_conflicts": lr["not_conflicts"], "exc_against": lr["exc_against"],
            "p2d": r["p2d"], "d2p": r["d2p"], "bma": r["bma"],
            "exact": self.scorer.exact_share(pat, prof),
            "size": np.log1p(prof.size.astype(np.float32)),
            "obs_log": lr["obs_log"],          # per observed term, per disease (2D)
            "ic_match": r["per_term_ic"],      # per observed term, per disease (2D)
        }

    def score(self, pat: Patient, pmid: str | None = None, strict: bool = False) -> Signals:
        arrays = self._score_block(pat, self.prof)
        q = self.query_tokens(pat)
        arrays["bm25"] = self.bm25.score(q).astype(np.float32)
        arrays["lsa"] = self.lsa.score(q).astype(np.float32)
        changed: list[str] = []
        strict_profile = None
        if strict and pmid:
            strict_profile = P.strict_profiles(self.strict, pmid, self.onto)
            if strict_profile is not None:
                changed = strict_profile.disease_ids
                block = self._score_block(pat, strict_profile)
                idx = np.array([self.row[d] for d in changed])
                for k, v in block.items():
                    if v.ndim == 2:
                        arrays[k][:, idx] = v
                    else:
                        arrays[k][idx] = v
                _, remaining = self.strict.affected(pmid)
                docs = build_disease_docs(self.t, annotations=remaining, disease_ids=changed)
                corpus = [docs[d] for d in changed]
                arrays["bm25"][idx] = self.bm25.score_new(q, corpus)
                arrays["lsa"][idx] = self.lsa.score_new(q, corpus)
        female = pat.sex == "FEMALE"
        arrays["xr_female"] = (self.xr & female).astype(np.float32)
        child = pat.age_years is not None and pat.age_years < 16
        arrays["adult_onset_child"] = (self.adult_only & child).astype(np.float32)
        return Signals(arrays=arrays, patient=pat, strict_diseases=changed, strict_profile=strict_profile)

    def profile_row(self, disease_id: str, sig: Signals | None = None):
        """The profile matrices and row for a disease, honouring strict removal."""
        if sig is not None and sig.strict_profile is not None and disease_id in sig.strict_profile.disease_ids:
            return sig.strict_profile, sig.strict_profile.disease_ids.index(disease_id)
        return self.prof, self.row[disease_id]

    def rank_of(self, scores: np.ndarray, disease_id: str) -> int:
        g = self.row[disease_id]
        return int((scores > scores[g]).sum()) + 1


def save_params(params: LRParams) -> None:
    (config.MODELS / "lr_params.json").write_text(json.dumps(params.__dict__, indent=2))


def load_params() -> LRParams:
    path = config.MODELS / "lr_params.json"
    if path.exists():
        return LRParams(**json.loads(path.read_text()))
    return LRParams()
