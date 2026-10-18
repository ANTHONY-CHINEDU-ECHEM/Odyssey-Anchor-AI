"""Input guard: redaction of identifiers and fixed paths for unsafe requests.

Redaction runs before anything is logged or shown back: NHS numbers, email
addresses, telephone numbers, UK postcodes, dates, hospital numbers and names
introduced by a title or a name field.

Fixed paths are decided by rules, not by a model, so they cannot be argued with:
  treatment     dosing, prescribing or management requests: diagnosis only
  override      attempts to change the instructions, in the question or hidden in a note
  crisis        a disclosure of self harm or suicidal thoughts
  emergency     signs of an acute emergency: the tool is not for urgent care
  certainty     demands for a definitive diagnosis: the answer stays a ranked differential
  off_topic     no phenotype, disease, gene or clinical term in the request
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REDACTIONS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("NHS_NUMBER", re.compile(r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+44\s?\d{2,4}|\b0\d{2,4})[\s-]?\d{3,4}[\s-]?\d{3,4}\b")),
    ("HOSPITAL_NUMBER", re.compile(r"\b(?:MRN|hospital (?:number|no\.?)|hosp\.? no\.?|patient id)\s*[:#]?\s*[A-Z0-9]{5,}\b", re.I)),
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")),
    ("DATE", re.compile(
        r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b", re.I)),
    ("NAME", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Master|Prof)\.?\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)?")),
    ("NAME", re.compile(r"\b(?:name|patient name|child's name)\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", re.I)),
]

TREATMENT = re.compile(
    r"\b(dose|doses|dosage|dosing|mg|mcg|prescribe|prescribing|prescription|treat|treated|treatment|"
    r"treatments|therapy|therapies|medication|medications|drug|drugs|manage|management|cure|surgery|"
    r"enzyme replacement|gene therapy)\b", re.I)
OVERRIDE = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules)|disregard (?:your|the) "
    r"(?:instructions|rules)|system prompt|you are now|developer mode|jailbreak|pretend (?:to be|you are)|"
    r"act as (?:a|an) (?:doctor|physician) and (?:prescribe|treat)|reveal your (?:prompt|instructions)", re.I)
CRISIS = re.compile(r"\b(suicid\w*|kill (?:myself|himself|herself)|end (?:my|his|her) life|self[- ]harm\w*|"
                    r"want to die|overdose on purpose)\b", re.I)
# Present tense signs of an emergency happening now; a history of status epilepticus is not one.
EMERGENCY = re.compile(r"\b(not breathing|cannot breathe|can't breathe|is unconscious|is unresponsive|"
                       r"unresponsive now|in anaphylaxis|anaphylaxis now|is in status epilepticus|"
                       r"seizing now|still seizing|cardiac arrest|bleeding heavily)\b", re.I)
CERTAINTY = re.compile(r"\b(confirm (?:the|a|that|this) diagnosis|definitely|for certain|for sure|"
                       r"tell me (?:exactly )?what (?:he|she|they|the patient) ha(?:s|ve)|which disease does .* have)\b", re.I)

MESSAGES = {
    "treatment": ("Odyssey Anchor AI supports differential diagnosis only. It does not advise on treatment, "
                  "dosing or management; those decisions belong to the treating team and the relevant specialist service."),
    "override": ("The request contained an instruction aimed at the system. It was ignored; the analysis "
                 "below follows the normal rules."),
    "crisis": ("If you or someone you are caring for is thinking about suicide or self harm, please contact "
               "emergency services or a crisis line now. In the UK, Samaritans can be reached on 116 123 at any time."),
    "emergency": ("This looks like an acute emergency. Odyssey Anchor AI is not for urgent care; follow local "
                  "emergency procedures now."),
    "certainty": ("A phenotype comparison cannot confirm a diagnosis. The answer is a ranked differential that "
                  "needs genetic testing and specialist review."),
    "off_topic": ("Odyssey Anchor AI answers questions about rare disease phenotypes, diseases and genes, and "
                  "analyses clinical descriptions. The request did not contain any of these."),
}


@dataclass
class GuardResult:
    text: str
    question: str = ""
    redactions: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f in ("crisis", "emergency") for f in self.flags)

    def messages(self) -> list[str]:
        return [MESSAGES[f] for f in self.flags if f in MESSAGES]


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pat in REDACTIONS:
        text, n = pat.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


def check(note: str = "", question: str = "", has_clinical_content: bool = True) -> GuardResult:
    joined = f"{note}\n{question}"
    red_note, counts = redact(note or "")
    red_q, counts_q = redact(question or "")
    for k, v in counts_q.items():
        counts[k] = counts.get(k, 0) + v
    flags = []
    if CRISIS.search(joined):
        flags.append("crisis")
    if EMERGENCY.search(joined):
        flags.append("emergency")
    if OVERRIDE.search(joined):
        flags.append("override")
    if question and TREATMENT.search(question):
        flags.append("treatment")
    if question and CERTAINTY.search(question):
        flags.append("certainty")
    if not has_clinical_content:
        flags.append("off_topic")
    text = red_note + ("\n" + red_q if red_q else "")
    return GuardResult(text=text, question=red_q, redactions=counts, flags=flags)
