"""Render a structured case as a clinical style note, for testing extraction.

No open corpus of clinical notes paired with HPO terms exists for these
patients, so the extraction evaluation renders each case into prose. The
renderer is built to be harder than a template list:

  phrasing     a synonym instead of the label for half of the features, and for
               "X of the Y" labels an inverted form ("corpus callosum hypoplasia")
  negation     eight different cue phrases, before or after the finding, with
               coordinated lists ("no seizures, ataxia or hypotonia")
  distractors  family history sentences naming real phenotypes the patient does
               not have, and normal findings that mention no phenotype
  noise        the age and sex sentence, order shuffled, mixed case

Results on rendered notes measure the recogniser's handling of vocabulary,
negation and family history; they do not measure performance on real notes.
"""
from __future__ import annotations

import random
import re

from ..ontology.graph import Ontology

NEG_PRE = ["{p} has no {x}", "{p} presented without {x}", "{p} denies {x}", "{p} was negative for {x}",
           "There was no evidence of {x}", "Absence of {x} was documented", "There is no history of {x}",
           "{p} has remained free of {x}"]
NEG_POST = ["was absent", "was not observed", "was ruled out", "was not present"]
OPENERS = ["On examination there was", "Clinically there was", "The history includes", "Findings included",
           "{p} was noted to have", "Assessment showed", "Further review identified"]
NORMALS = ["Vital signs were within normal limits.", "The remainder of the examination was unremarkable.",
           "Routine biochemistry was reported as normal.", "Growth parameters were reviewed at each visit.",
           "Immunisations are up to date."]
RELATIVES = ["mother", "father", "older brother", "maternal aunt", "paternal grandfather", "sister"]
OF_THE = re.compile(r"^(?P<head>[A-Za-z ]+?) of (the )?(?P<tail>[A-Za-z ]+)$")


def phrase(onto: Ontology, tid: str, rng: random.Random) -> str:
    i = onto.index[tid]
    label = onto.labels[i]
    syns = [s for s in onto.synonyms[i] if 3 <= len(s) <= 60 and not s.isupper()]
    text = rng.choice(syns) if syns and rng.random() < 0.5 else label
    m = OF_THE.match(text)
    if m and rng.random() < 0.5 and len(m.group("tail").split()) <= 3:
        text = f"{m.group('tail')} {m.group('head').lower()}"
    return text if rng.random() < 0.5 else text.lower()


def _list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _or_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " or " + items[-1]


def render(onto: Ontology, observed: list[str], excluded: list[str], sex: str, age: float | None,
           seed: int, distractor_pool: list[str] | None = None, max_excluded: int = 8) -> dict:
    rng = random.Random(seed)
    obs = [t for t in observed if t in onto.index]
    exc = [t for t in excluded if t in onto.index][:max_excluded]
    rng.shuffle(obs)
    rng.shuffle(exc)
    pron = "She" if sex == "FEMALE" else ("He" if sex == "MALE" else "The patient")
    who = "girl" if sex == "FEMALE" else ("boy" if sex == "MALE" else "child")
    sentences = []
    if age is not None and age == age:
        years = int(age)
        sentences.append(f"This {years} year old {who if years < 18 else 'adult'} was referred for genetic assessment.")
    else:
        sentences.append("The patient was referred for genetic assessment.")
    k = 0
    while k < len(obs):
        chunk = obs[k:k + rng.choice([1, 2, 3])]
        k += len(chunk)
        sentences.append(f"{rng.choice(OPENERS).format(p=pron)} {_list([phrase(onto, t, rng) for t in chunk])}.")
    k = 0
    while k < len(exc):
        chunk = exc[k:k + rng.choice([1, 2, 3])]
        k += len(chunk)
        words = [phrase(onto, t, rng) for t in chunk]
        if len(chunk) == 1 and rng.random() < 0.4:
            sentences.append(f"{words[0][0].upper() + words[0][1:]} {rng.choice(NEG_POST)}.")
        else:
            sentences.append(rng.choice(NEG_PRE).format(p=pron, x=_or_list(words)) + ".")
    family_terms = []
    if distractor_pool:
        pool = [t for t in distractor_pool if t not in obs and t not in exc]
        for t in rng.sample(pool, k=min(2, len(pool))):
            family_terms.append(t)
            sentences.append(f"Family history: the {rng.choice(RELATIVES)} has {phrase(onto, t, rng)}.")
    sentences.append(rng.choice(NORMALS))
    head, body = sentences[0], sentences[1:]
    rng.shuffle(body)
    return {"text": " ".join([head] + body), "observed": obs, "excluded": exc, "family": family_terms}
