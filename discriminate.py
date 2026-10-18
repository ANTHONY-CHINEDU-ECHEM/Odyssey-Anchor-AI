"""Next best question: the phenotype whose answer would most sharpen the differential.

Given the posterior over the leading candidates, every askable phenotype q is
scored by its expected information gain:

  EIG(q) = H(pi) - [ P(yes) H(pi | yes) + P(no) H(pi | no) ]

with P(q | d) taken from the propagated annotation frequency and smoothed to
the background. Askable means annotated directly to at least one leading
candidate, specific enough to be a real clinical question, and not already
implied by or contradicting what is recorded. The output also says which
candidate a yes and a no would favour, so the clinician sees why it matters.
"""
from __future__ import annotations

import numpy as np

from ..rank.engine import Engine, Signals


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


def next_questions(eng: Engine, sig: Signals, candidates: list[str], posterior: np.ndarray,
                   k: int = 5, min_ic: float = 2.5, eta: float = 0.02) -> list[dict]:
    onto, ic, bg = eng.onto, eng.ic, eng.bg
    pi = np.asarray(posterior, dtype=np.float64)
    pi = pi / pi.sum()
    rows = [eng.profile_row(d, sig) for d in candidates]
    pat = sig.patient
    known = set(pat.closure.tolist())
    blocked = set(known)
    for e in pat.excluded:
        blocked.update(onto.ancestor_lists[e].tolist())
    descendants_of_known = set()
    pool = set()
    for prof, r in rows:
        lo, hi = prof.DIR.indptr[r], prof.DIR.indptr[r + 1]
        pool.update(prof.DIR.indices[lo:hi].tolist())
    cand = []
    for q in pool:
        if q in blocked or ic[q] < min_ic or not onto.phenotype_mask[q]:
            continue
        # Skip terms below an observed term: they refine, rarely discriminate, and read oddly.
        if any(int(a) in known for a in onto.ancestor_lists[q] if a != q and ic[a] > 3.0):
            descendants_of_known.add(q)
            continue
        cand.append(q)
    if not cand:
        return []
    cand = np.array(sorted(cand))
    like = np.zeros((len(cand), len(candidates)))
    for j, (prof, r) in enumerate(rows):
        row = prof.F[r]
        f = dict(zip(row.indices.tolist(), row.data.tolist()))
        like[:, j] = [f.get(int(q), 0.0) for q in cand]
    like = (1 - eta) * like + eta * bg[cand][:, None]
    p_yes = like @ pi
    post_yes = like * pi / p_yes[:, None]
    post_no = (1 - like) * pi / (1 - p_yes)[:, None]
    h0 = entropy(pi)
    eig = h0 - (p_yes * entropy(post_yes) + (1 - p_yes) * entropy(post_no))
    order = np.argsort(-eig)[:k]
    out = []
    for i in order:
        q = int(cand[i])
        yes_top = int(np.argmax(post_yes[i]))
        no_top = int(np.argmax(post_no[i]))
        out.append({
            "term": onto.ids[q], "label": onto.labels[q], "expected_gain_bits": round(float(eig[i]), 3),
            "p_yes": round(float(p_yes[i]), 3),
            "if_present_favours": candidates[yes_top], "if_present_probability": round(float(post_yes[i, yes_top]), 3),
            "if_absent_favours": candidates[no_top], "if_absent_probability": round(float(post_no[i, no_top]), 3),
            "seen_in": [candidates[j] for j in np.argsort(-like[i])[:3] if like[i, j] > 0.3],
        })
    return out
