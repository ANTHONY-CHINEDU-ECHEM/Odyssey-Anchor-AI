"""Case perturbations for robustness testing and the end to end note pathway.

Every transform is a picklable callable taking (case, rng) and returning a new
case. They run inside the scoring workers and reach the worker's engine
through runner._W, so the ontology is never copied between processes.
"""
from __future__ import annotations

import random

from . import runner


class DropHalf:
    name = "drop half of the observed features"

    def __call__(self, case: dict, rng: random.Random) -> dict:
        obs = list(case["observed"])
        rng.shuffle(obs)
        case["observed"] = obs[: max(1, len(obs) // 2)]
        return case


class KeepThree:
    name = "keep three observed features"

    def __call__(self, case: dict, rng: random.Random) -> dict:
        obs = list(case["observed"])
        rng.shuffle(obs)
        case["observed"] = obs[:3]
        return case


class NoExcluded:
    name = "drop every excluded feature"

    def __call__(self, case: dict, rng: random.Random) -> dict:
        case["excluded"] = []
        return case


class Generalise:
    name = "replace every observed feature with a parent term"

    def __call__(self, case: dict, rng: random.Random) -> dict:
        onto = runner._W["eng"].onto
        out = []
        for t in case["observed"]:
            i = onto.index.get(t)
            if i is None or not onto.parents[i]:
                out.append(t)
                continue
            p = rng.choice(onto.parents[i])
            out.append(onto.ids[p] if onto.phenotype_mask[p] else t)
        case["observed"] = out
        return case


class AddNoise:
    name = "add two unrelated common features"

    def __call__(self, case: dict, rng: random.Random) -> dict:
        eng = runner._W["eng"]
        if "noise_pool" not in runner._W:
            # Common, mid specificity phenotypes: realistic incidental findings.
            ic = eng.ic
            pool = [i for i in range(eng.onto.n) if eng.onto.phenotype_mask[i] and 3.0 <= ic[i] <= 5.0]
            runner._W["noise_pool"] = pool
        pool = runner._W["noise_pool"]
        closure = set(eng.onto.closure(eng.onto.idx(case["observed"])).tolist())
        added = []
        while len(added) < 2:
            c = rng.choice(pool)
            if c not in closure and not closure.intersection(eng.onto.ancestor_lists[c].tolist()[-3:]):
                added.append(eng.onto.ids[c])
        case["observed"] = list(case["observed"]) + added
        return case


class FromNote:
    """Render the case as a clinical note, then read the phenotypes back with the recogniser."""
    name = "rendered note, then extraction"

    def __call__(self, case: dict, rng: random.Random) -> dict:
        from ..extract.recognizer import Recognizer
        from ..extract.render import render
        eng = runner._W["eng"]
        if "recognizer" not in runner._W:
            runner._W["recognizer"] = Recognizer(eng.onto, eng.ic)
            ic = eng.ic
            runner._W["distractors"] = [eng.onto.ids[i] for i in range(eng.onto.n)
                                        if eng.onto.phenotype_mask[i] and 3.0 <= ic[i] <= 6.0][::7]
        rec = runner._W["recognizer"]
        note = render(eng.onto, case["observed"], case["excluded"], case.get("sex", "UNKNOWN_SEX"),
                      case.get("age_years"), seed=rng.randint(0, 10**9), distractor_pool=runner._W["distractors"])
        terms = rec.to_terms(rec.extract(note["text"]))
        case["observed"], case["excluded"] = terms["observed"], terms["excluded"]
        if not case["observed"]:
            return None
        return case


TRANSFORMS = {"drop_half": DropHalf(), "keep_three": KeepThree(), "no_excluded": NoExcluded(),
              "generalise": Generalise(), "add_noise": AddNoise(), "from_note": FromNote()}
