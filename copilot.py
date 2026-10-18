"""The copilot: a state machine of eleven timed nodes.

  guard        redact identifiers, raise fixed path flags
  extract      read phenotypes from free text (present, absent, family, uncertain)
  normalise    map terms to current HPO identifiers, keep the most specific
  score        every base ranker over all 12,880 diseases
  fuse         shortlist, fused ranker, temperature calibrated posterior
  group        merge OMIM and Orphanet records of one Mondo concept
  explain      evidence subgraph for each leading candidate
  discriminate next best questions by expected information gain
  genes        gene panel behind the leading candidates
  retrieve     knowledge passages for a free text question
  compose      cited sentences
  verify       every sentence against its sources
  audit        hash chained record of the run

Every node is timed and its output kept, so the interface can show exactly how
an answer was produced.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..extract.recognizer import Recognizer
from ..knowledge.build import load_tables
from ..rank.engine import Engine, load_params
from ..rank.features import feature_frame
from ..rank.fusion import load_product
from ..reason.discriminate import next_questions
from ..reason.evidence import AnnotationLookup, explain
from ..reason.genes import GeneIndex
from ..retrieval import passages as passage_index
from . import guard as guard_mod
from .audit import AuditLog
from .composer import compose_answer, compose_differential, model_source, patient_source
from . import writer_llm
from .verifier import check_sentence, verify


LEADING = re.compile(r"\b(leading|top|first|most likely) (candidate|diagnosis|disease)\b", re.I)


@dataclass
class Request:
    note: str = ""
    observed: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    question: str = ""
    sex: str = "UNKNOWN_SEX"
    age_years: float | None = None
    strict_pmid: str | None = None     # evaluation and teaching cases: remove the source paper
    context: list[str] = field(default_factory=list)
    log: bool = True


class Copilot:
    def __init__(self, tables: dict[str, pd.DataFrame] | None = None, audit: AuditLog | None = None):
        self.t = tables or load_tables()
        self.engine = Engine(self.t, params=load_params())
        self.recognizer = Recognizer(self.engine.onto, self.engine.ic)
        self.model, self.meta = load_product()
        self.T = float(self.meta["temperature"])
        self.lookup = AnnotationLookup(self.t["annotations"])
        self.genes = GeneIndex(self.t["disease_genes"])
        self.gene_set = set(self.t["disease_genes"].gene_symbol)
        d = self.t["diseases"].set_index("disease_id")
        self.names = d["name"].to_dict()
        self.mondo = d["mondo_id"].to_dict()
        self.mondo_label = d["mondo_label"].to_dict()
        self.inheritance = d["inheritance"].to_dict()
        self.dgenes = d["genes"].to_dict()
        self.passages = passage_index.load(self.engine.onto, self.recognizer.lex)
        self.ptext = dict(zip(self.passages.p.pid, self.passages.p.title + ". " + self.passages.p.text))
        self.audit = audit or AuditLog()

    # nodes -----------------------------------------------------------------------
    def _terms(self, ids: list[str]) -> list[dict]:
        onto = self.engine.onto
        return [{"id": t, "label": onto.labels[onto.index[t]]} for t in ids if t in onto.index]

    def differential(self, sig, k: int = config.MAX_DIFFERENTIAL) -> tuple[list[dict], pd.DataFrame]:
        feats = feature_frame(self.engine, sig)
        feats["score"] = self.model.score(feats)
        z = feats.score.to_numpy() / self.T
        z = np.exp(z - z.max())
        feats["prob"] = z / z.sum()
        feats["concept"] = [self.mondo.get(d) if isinstance(self.mondo.get(d), str) else d for d in feats.disease_id]
        grouped = []
        for concept, g in feats.groupby("concept"):
            g = g.sort_values("score", ascending=False)
            grouped.append({"concept": concept, "disease_id": g.disease_id.iloc[0], "score": float(g.score.iloc[0]),
                            "probability": float(g.prob.sum()), "members": g.disease_id.tolist()})
        # A concept's probability is the sum over its OMIM and Orphanet records; rank by it.
        grouped.sort(key=lambda x: (-x["probability"], -x["score"]))
        out = []
        for i, c in enumerate(grouped[:k]):
            did = c["disease_id"]
            row = self.engine.row[did]
            genes = sorted({g for m in c["members"] for g in self.dgenes.get(m, [])})
            inheritance = sorted({x for m in c["members"] for x in self.inheritance.get(m, [])})
            label = self.mondo_label.get(did)
            out.append({
                "rank": i + 1, **c, "name": self.names.get(did, did),
                "mondo_label": label if isinstance(label, str) else None,
                "inheritance": inheritance, "genes": genes,
                "base_ranks": {k: int((sig[k] > sig[k][row]).sum()) + 1 for k in ("lr", "bma", "bm25", "lsa")},
            })
        return out, feats

    def _llm_answer(self, question: str, hits, fallback: list) -> tuple[list, bool]:
        """Ask the optional writer; keep its sentences only if at least half pass the verifier."""
        srcs = {r.pid: self.ptext.get(r.pid, r.text) for r in hits.head(4).itertuples()}
        try:
            drafted = writer_llm.write(question, srcs)
        except Exception:
            return fallback, False
        checked = [check_sentence(s, srcs, self.gene_set) for s in drafted]
        kept = [s for s in checked if s.passed]
        if drafted and len(kept) * 2 >= len(drafted):
            for s in kept:
                s.reasons = []
            return kept, True
        return fallback, False

    def run(self, req: Request) -> dict:
        timings, t0 = {}, time.perf_counter()

        def tick(name: str) -> None:
            nonlocal t0
            now = time.perf_counter()
            timings[name] = round((now - t0) * 1000, 2)
            t0 = now

        run_id = uuid.uuid4().hex[:12]
        g = guard_mod.check(req.note, req.question, has_clinical_content=True)
        tick("guard")
        findings = self.recognizer.extract(g.text) if req.note else []
        found = self.recognizer.to_terms(findings)
        tick("extract")
        observed = list(dict.fromkeys(list(req.observed) + found["observed"]))
        excluded = list(dict.fromkeys(list(req.excluded) + found["excluded"]))
        pat = self.engine.patient(observed, excluded, req.sex, req.age_years)
        tick("normalise")

        result = {"run_id": run_id, "guard": {"flags": g.flags, "redactions": g.redactions, "messages": g.messages()},
                  "findings": self.recognizer.as_dicts(findings),
                  "patient": {"observed": self._terms([self.engine.onto.ids[i] for i in pat.observed]),
                              "excluded": self._terms([self.engine.onto.ids[i] for i in pat.excluded]),
                              "uncertain": self._terms(found["uncertain"]), "family": self._terms(found["family"]),
                              "sex": req.sex, "age_years": req.age_years},
                  "differential": [], "questions": [], "genes": [], "answer": None}
        sentences, sources = [], {}
        has_phenotypes = len(pat.observed) > 0
        if g.blocked:
            tick("score")
        elif has_phenotypes:
            sig = self.engine.score(pat, pmid=req.strict_pmid, strict=req.strict_pmid is not None)
            tick("score")
            diff, _ = self.differential(sig)
            tick("fuse")
            tick("group")
            for d in diff:
                d["evidence"] = explain(self.engine, sig, d["disease_id"], self.lookup, drop_ref=req.strict_pmid)
            tick("explain")
            post = np.array([d["probability"] for d in diff])
            questions = next_questions(self.engine, sig, [d["disease_id"] for d in diff], post) if diff else []
            tick("discriminate")
            panel = self.genes.panel_concepts([(d["members"], d["probability"]) for d in diff], self.names)
            tick("genes")
            result.update({"differential": diff, "questions": questions, "genes": panel})
            sources["PATIENT"] = patient_source(result["patient"]["observed"], result["patient"]["excluded"])
            sources["MODEL"] = model_source(diff, questions, self.names)
            sentences += compose_differential(diff, questions, panel, len(pat.observed), len(pat.excluded), self.names)
        else:
            for name in ("score", "fuse", "group", "explain", "discriminate", "genes"):
                tick(name)

        if req.question and not g.blocked:
            diff = result["differential"]
            if req.context:
                ctx = req.context
            elif LEADING.search(req.question):
                ctx = diff[0]["members"] if diff else []
            else:
                ctx = [m for d in diff[:5] for m in d["members"]]
            hits, info = self.passages.search(g.question, k=5, context=ctx)
            tick("retrieve")
            ents = info["diseases"] or info["genes"] or info["terms"]
            if "treatment" in g.flags:
                result["answer"] = {"intent": info["intent"], "passages": [], "entities": info}
            elif not ents and not has_phenotypes:
                if "off_topic" not in g.flags:
                    g.flags.append("off_topic")
                result["answer"] = {"intent": info["intent"], "passages": [], "entities": info}
            else:
                for r in hits.itertuples():
                    sources[r.pid] = self.ptext.get(r.pid, r.text)
                result["answer"] = {"intent": info["intent"], "entities": info,
                                    "passages": hits[["pid", "kind", "title", "text", "score"]].to_dict("records")}
                onto = self.engine.onto
                answer = compose_answer(req.question, hits, info,
                                        label_of=lambda t: onto.labels[onto.index[t]] if t in onto.index else t)
                result["answer"]["writer"] = "deterministic"
                if writer_llm.enabled():
                    answer, used = self._llm_answer(req.question, hits, answer)
                    result["answer"]["writer"] = "language model" if used else "deterministic (model output failed verification)"
                sentences += answer
        else:
            tick("retrieve")
        tick("compose")
        for s in sentences:
            for c in s.cites:
                if c not in sources and c in self.ptext:
                    sources[c] = self.ptext[c]
        v = verify(sentences, sources, self.gene_set)
        tick("verify")
        result["sentences"] = [s for s in v["sentences"] if s["passed"]]
        result["verification"] = {"passed": v["passed"], "total": v["total"], "dropped": v["dropped"]}
        result["guard"]["flags"] = g.flags
        result["guard"]["messages"] = g.messages()
        result["sources"] = {k: v for k, v in sources.items()}
        if req.log:
            result["audit_hash"] = self.audit.append({
                "run_id": run_id, "input": g.text[:4000], "flags": g.flags, "redactions": g.redactions,
                "observed": [t["id"] for t in result["patient"]["observed"]],
                "excluded": [t["id"] for t in result["patient"]["excluded"]],
                "differential": [(d["disease_id"], round(d["probability"], 4)) for d in result["differential"]],
                "verified": f"{v['passed']}/{v['total']}", "strict_pmid": req.strict_pmid,
            })
        tick("audit")
        result["timings_ms"] = timings
        return result
