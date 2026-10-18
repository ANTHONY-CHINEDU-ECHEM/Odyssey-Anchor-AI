"""Sentence verifier: nothing reaches the clinician unless its sources support it.

Every sentence carries citations. A sentence passes only if all checks pass:

  cited          it cites at least one source (fixed policy sentences excepted)
  exists         every cited source exists: a knowledge base passage, the patient
                 record (PATIENT) or the computed model output (MODEL)
  lexical        at least 40% of its content words appear in the cited sources
  numbers        every number appears in the cited sources
  identifiers    every disease, phenotype or Mondo identifier appears in the cited sources
  genes          every gene symbol appears in the cited sources
  no_treatment   no treatment, dosing or management language
  no_certainty   no claim that the patient has a disease or that a diagnosis is confirmed

The verifier checks words and numbers, not meaning: a sentence that recombines
supported words into a wrong claim can pass. That limit is measured and
reported in the evaluation.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .. import config
from ..retrieval.text import tokens

NUMBER = re.compile(r"(?<![A-Za-z:])\d+(?:\.\d+)?")
IDENT = re.compile(r"\b(?:OMIM|ORPHA|DECIPHER|MONDO|HP):\d+\b")
DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|units?|iu|IU)(?:/kg)?\b")
TREATMENT = re.compile(
    r"\b(treat|treated|treating|treatment|treatments|therapy|therapies|prescribe|prescribed|dose|doses|dosage|"
    r"medication|medications|drug|drugs|management|manage|surgery|surgical|cure|transplant\w*|supplement\w*)\b", re.I)
CERTAINTY = re.compile(
    r"\b(?:the patient|he|she|they|this child|this patient) (?:has|have|is suffering from)\b(?! (?:been )?recorded)"
    r"|\bdiagnosed with\b|\bconfirm(?:s|ed)? (?:the |a )?diagnosis\b|\bdefinitely\b|\bcertainly\b|\bthe diagnosis is\b", re.I)

# Words the composer adds around cited content; excluded from the lexical support check.
GLUE = set(tokens(
    "leading candidate candidates estimated probability among shortlisted supporting features feature disease "
    "annotated recorded absent typical worth checking next informative question whether present favours favour "
    "would most separate genes behind order listed knowledge base passage reports reported describes describe "
    "consistent phenotype record matches match match through shared ancestor more specific general term "
    "recorded features absences inheritance onset linked from according answer this following overview "
    "patient analysis analysed extracted note terms term observed excluded frequency stated rank ranked "
    "concept concepts belongs group explain explained explains also known"))

DECISION_SUPPORT = ("This is decision support for a clinical geneticist: any diagnosis needs genetic testing "
                    "and specialist review.")
NO_PASSAGE = "No passage in the knowledge base answers this question."
POLICY = (DECISION_SUPPORT, NO_PASSAGE)


@dataclass
class Sentence:
    text: str
    cites: list[str]
    role: str
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_sentence(s: Sentence, sources: dict[str, str], genes: set[str]) -> Sentence:
    reasons = []
    if s.role == "policy":
        if s.text not in POLICY:
            reasons.append("policy text not on the fixed list")
    else:
        if not s.cites:
            reasons.append("cited")
        missing = [c for c in s.cites if c not in sources]
        if missing:
            reasons.append("exists")
        support_text = " ".join(sources.get(c, "") for c in s.cites)
        support_tokens = set(tokens(support_text))
        content = [t for t in tokens(s.text) if t not in GLUE and not t.isdigit()]
        if content:
            share = sum(t in support_tokens for t in content) / len(content)
            if share < config.MIN_LEXICAL_SUPPORT:
                reasons.append("lexical")
        support_numbers = set(NUMBER.findall(support_text))
        nums = [n for n in NUMBER.findall(re.sub(IDENT, "", s.text))]
        if any(n not in support_numbers for n in nums):
            reasons.append("numbers")
        if any(i not in support_text for i in IDENT.findall(s.text)):
            reasons.append("identifiers")
        mentioned = {w for w in re.findall(r"\b[A-Z][A-Z0-9-]{1,9}\b", s.text) if w in genes}
        if any(g not in support_text for g in mentioned):
            reasons.append("genes")
    if DOSE.search(s.text) or TREATMENT.search(s.text):
        reasons.append("no_treatment")
    if CERTAINTY.search(s.text):
        reasons.append("no_certainty")
    s.passed = not reasons
    s.reasons = reasons
    return s


def verify(sentences: list[Sentence], sources: dict[str, str], genes: set[str]) -> dict:
    checked = [check_sentence(s, sources, genes) for s in sentences]
    return {
        "sentences": [s.to_dict() for s in checked],
        "passed": sum(s.passed for s in checked),
        "total": len(checked),
        "dropped": [s.to_dict() for s in checked if not s.passed],
    }
