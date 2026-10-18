"""Clinical phenotype recognition from free text.

1. Every HPO label, exact synonym, plural form and layperson synonym enters a
   longest match lexicon (order insensitive for short phrases).
2. Text is split into sentences and clauses. Each match is classified by the
   cues in its clause:
     negated     no, not, without, denies, absence of, negative for, ruled out,
                 free of, never had; or after the mention: was absent, not seen
     family      mother, father, sibling, aunt, family history ...
     uncertain   possible, suspected, query, may have, questionable ...
   Negation scope runs to the end of the clause, so "no seizures, hypotonia or
   ataxia" negates all three, and stops at but, however, although, except.
3. Overlapping candidate concepts are resolved by preferring a primary label,
   then an exact synonym, then the more specific term.

The output keeps the character span and the cue for every finding, so the
interface can highlight exactly why a term was read as present or absent.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from ..ontology.graph import Ontology
from ..retrieval.text import Lexicon

PRE_NEG = re.compile(
    r"\b(no|not|without|denies|denied|denying|absence of|absent|negative for|ruled out|rules out|"
    r"free of|never had|never developed|lack of|lacks|lacking|did not have|does not have|"
    r"did not show|does not show|no evidence of|no history of|no signs of|nor|neither)\b", re.I)
POST_NEG = re.compile(
    r"^\W*(was|were|is|are|has been|have been)?\s*(absent|not seen|not observed|not present|"
    r"excluded|ruled out|not detected|not found|not noted|negative)\b", re.I)
FAMILY = re.compile(
    r"\b(mother|father|parent|parents|sibling|siblings|brother|sister|sisters|brothers|aunt|uncle|cousin|"
    r"grandmother|grandfather|maternal|paternal|family history|relative|relatives|twin)\b", re.I)
UNCERTAIN = re.compile(
    r"\b(possible|possibly|suspected|suspect|query|questionable|may have|might have|probable|"
    r"cannot exclude|could not exclude|unclear whether|equivocal)\b|\?", re.I)
CLAUSE_BREAK = re.compile(r"\b(but|however|although|though|except|whereas|while)\b|[;:]", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Self references reset family scope: "her mother has X; she has Y"
SELF = re.compile(r"\b(the patient|patient|proband|he|she|they|the child|the boy|the girl|index case)\b", re.I)


@dataclass
class Finding:
    hpo_id: str
    label: str
    text: str
    start: int
    end: int
    status: str        # present, absent, family, uncertain
    cue: str
    match: str         # exact, reordered
    via: str           # label or exact synonym (layperson synonyms are included as synonyms)


class Recognizer:
    def __init__(self, onto: Ontology, ic: np.ndarray | None = None):
        self.onto = onto
        self.ic = ic if ic is not None else onto.depth.astype(np.float32)
        self.lex = Lexicon(max_len=10)
        self.kind: dict[tuple[str, str], str] = {}
        for i, tid in enumerate(onto.ids):
            if not onto.phenotype_mask[i]:
                continue
            self._add(onto.labels[i], tid, "label")
            for s in onto.synonyms[i]:
                self._add(s, tid, "exact synonym")

    def _add(self, phrase: str, tid: str, via: str) -> None:
        phrase = phrase.strip()
        if len(phrase) < 3:
            return
        self.lex.add(phrase, tid)
        key = (phrase.lower(), tid)
        rank = {"label": 0, "exact synonym": 1}
        if key not in self.kind or rank[via] < rank[self.kind[key]]:
            self.kind[key] = via

    def _choose(self, text: str, concepts: list[str]) -> tuple[str, str]:
        best, best_key = concepts[0], None
        for c in concepts:
            via = self.kind.get((text.lower(), c), "exact synonym")
            order = {"label": 0, "exact synonym": 1}[via]
            key = (order, -float(self.ic[self.onto.index[c]]))
            if best_key is None or key < best_key:
                best, best_key = c, key
        via = self.kind.get((text.lower(), best), "exact synonym")
        return best, via

    def extract(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        pos = 0
        for sentence in SENTENCE.split(text):
            if not sentence.strip():
                pos += len(sentence) + 1
                continue
            offset = text.find(sentence, pos)
            offset = pos if offset < 0 else offset
            pos = offset + len(sentence)
            # Clause boundaries within the sentence.
            bounds = [0] + [m.start() for m in CLAUSE_BREAK.finditer(sentence)] + [len(sentence)]
            for m in self.lex.find(sentence):
                cstart = max(b for b in bounds if b <= m["start"])
                cend = min(b for b in bounds if b >= m["end"])
                before = sentence[cstart:m["start"]]
                after = sentence[m["end"]:cend]
                status, cue = "present", ""
                fam = list(FAMILY.finditer(sentence[:m["start"]]))
                selfref = list(SELF.finditer(sentence[:m["start"]]))
                if fam and (not selfref or selfref[-1].start() < fam[-1].start()):
                    status, cue = "family", fam[-1].group()
                neg = list(PRE_NEG.finditer(before))
                post = POST_NEG.search(after)
                if status == "present" and (neg or post):
                    status, cue = "absent", (neg[-1].group() if neg else post.group().strip())
                if status == "present":
                    unc = UNCERTAIN.search(before) or UNCERTAIN.search(after[:12])
                    if unc:
                        status, cue = "uncertain", unc.group()
                tid, via = self._choose(m["text"], m["concepts"])
                i = self.onto.index[tid]
                findings.append(Finding(
                    hpo_id=tid, label=self.onto.labels[i], text=m["text"],
                    start=offset + m["start"], end=offset + m["end"], status=status, cue=cue,
                    match=m["match"], via=via,
                ))
        return findings

    def to_terms(self, findings: list[Finding]) -> dict[str, list[str]]:
        present = [f.hpo_id for f in findings if f.status == "present"]
        absent = [f.hpo_id for f in findings if f.status == "absent" and f.hpo_id not in present]
        return {
            "observed": list(dict.fromkeys(present)),
            "excluded": list(dict.fromkeys(absent)),
            "uncertain": list(dict.fromkeys(f.hpo_id for f in findings if f.status == "uncertain")),
            "family": list(dict.fromkeys(f.hpo_id for f in findings if f.status == "family")),
        }

    @staticmethod
    def as_dicts(findings: list[Finding]) -> list[dict]:
        return [asdict(f) for f in findings]
