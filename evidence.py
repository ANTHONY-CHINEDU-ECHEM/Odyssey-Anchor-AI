"""Evidence subgraph for one candidate disease.

For every observed patient term the explainer finds the disease annotation that
best accounts for it and the path between them in the ontology:

  exact            the disease is annotated with the same term
  more specific    the patient term sits below the disease annotation
                   (patient: focal seizure; disease: seizure)
  more general     the disease annotation sits below the patient term
                   (patient: seizure; disease: focal seizure)
  related          the two meet at a shared ancestor, named in the output
  unexplained      the best shared ancestor is uninformative

It then lists contradictions (features recorded absent that the disease shows
in at least 30% of patients, and explicit NOT annotations) and the most
characteristic features of the disease that the record does not mention,
which are the natural things to look for next. Every item names the knowledge
base passage that supports it, so the composer can cite it.
"""
from __future__ import annotations

import numpy as np

from ..knowledge.ids import disease_overview_id, disease_system_id
from ..rank.engine import Engine, Signals

SUPPORT, AGAINST = np.log(2.0), np.log(0.5)


class AnnotationLookup:
    """(disease, term) to frequency phrase and references."""

    def __init__(self, annotations):
        pheno = annotations[annotations.aspect.eq("P") & ~annotations.negated]
        self.map = {}
        for r in pheno.itertuples():
            key = (r.disease_id, r.hpo_id)
            prev = self.map.get(key)
            if prev is None or r.freq > prev[0]:
                self.map[key] = (float(r.freq), r.frequency, list(r.refs))

    def get(self, disease_id: str, hpo_id: str, drop_ref: str | None = None):
        v = self.map.get((disease_id, hpo_id))
        if v is None:
            return None
        refs = [x for x in v[2] if x != drop_ref]
        return {"freq": v[0], "frequency": v[1], "refs": refs}


def explain(eng: Engine, sig: Signals, disease_id: str, lookup: AnnotationLookup,
            drop_ref: str | None = None) -> dict:
    onto, ic = eng.onto, eng.ic
    prof, row = eng.profile_row(disease_id, sig)
    g = eng.row[disease_id]
    pat = sig.patient
    lo, hi = prof.DIR.indptr[row], prof.DIR.indptr[row + 1]
    S = prof.DIR.indices[lo:hi]
    S_freq = prof.DIR.data[lo:hi]
    anc = onto.ancestor_lists
    matches = []
    obs_log = sig["obs_log"]
    for k, t in enumerate(pat.observed):
        mask = np.zeros(onto.n, dtype=bool)
        mask[anc[t]] = True
        best = (-1.0, None, None)
        for s in S:
            common = anc[s][mask[anc[s]]]
            if len(common) == 0:
                continue
            a = int(common[np.lexsort((onto.depth[common], ic[common]))[-1]])
            if ic[a] > best[0]:
                best = (float(ic[a]), int(s), a)
        ic_a, s, a = best
        lr = float(obs_log[k, g]) if obs_log.shape[0] else 0.0
        if s is None or ic_a < 1.0:
            kind = "unexplained"
        elif s == t:
            kind = "exact"
        elif mask[s]:
            kind = "more specific"
        elif t in set(anc[s].tolist()):
            kind = "more general"
        else:
            kind = "related"
        item = {
            "patient_term": onto.ids[t], "patient_label": onto.labels[t],
            "kind": kind, "log_lr": round(lr, 3),
            "verdict": "supports" if lr > SUPPORT else ("against" if lr < AGAINST else "neutral"),
            "information_content": round(float(ic[t]), 3),
        }
        if s is not None and kind != "unexplained":
            ann = lookup.get(disease_id, onto.ids[s], drop_ref) or {}
            item.update({
                "disease_term": onto.ids[s], "disease_label": onto.labels[s],
                "via_term": onto.ids[a], "via_label": onto.labels[a],
                "frequency": ann.get("frequency", ""), "freq": round(float(S_freq[list(S).index(s)]), 3),
                "refs": ann.get("refs", [])[:4],
                "passage": disease_system_id(disease_id, onto.systems[s]) if onto.systems[s] else disease_overview_id(disease_id),
            })
        matches.append(item)

    # Contradictions: excluded features the disease commonly shows.
    F_row = prof.F[row]
    f_cols = dict(zip(F_row.indices.tolist(), F_row.data.tolist()))
    contradictions = []
    for e in pat.excluded:
        f = f_cols.get(int(e), 0.0)
        if f >= 0.3:
            # Which annotation drives it: the most frequent annotated descendant of e.
            below = [s for s in S if e in set(anc[s].tolist())]
            src = max(below, key=lambda s: S_freq[list(S).index(s)]) if below else None
            ann = lookup.get(disease_id, onto.ids[src], drop_ref) if src is not None else None
            contradictions.append({
                "excluded_term": onto.ids[e], "excluded_label": onto.labels[e], "freq": round(f, 3),
                "disease_term": onto.ids[src] if src is not None else onto.ids[e],
                "disease_label": onto.labels[src] if src is not None else onto.labels[e],
                "frequency": (ann or {}).get("frequency", ""),
                "passage": disease_system_id(disease_id, onto.systems[src]) if src is not None and onto.systems[src] else disease_overview_id(disease_id),
            })
    not_lo, not_hi = prof.NOT.indptr[row], prof.NOT.indptr[row + 1]
    not_terms = set(prof.NOT.indices[not_lo:not_hi].tolist())
    for t in pat.observed:
        hit = not_terms.intersection(anc[t].tolist())
        for n_ in hit:
            contradictions.append({
                "observed_term": onto.ids[t], "observed_label": onto.labels[t], "not_annotation": onto.ids[n_],
                "not_label": onto.labels[n_], "freq": 0.0,
                "passage": disease_system_id(disease_id, onto.systems[n_]) if onto.systems[n_] else disease_overview_id(disease_id),
            })

    # Characteristic features not yet recorded either way.
    known = set(pat.closure.tolist())
    for e in pat.excluded:
        known.update(anc[e].tolist())
    related_down = set()
    for t in list(pat.observed) + list(pat.excluded):
        related_down.add(int(t))
    to_check = []
    for s, f in zip(S, S_freq):
        if f < 0.4 or ic[s] < 3.0 or int(s) in known:
            continue
        if any(int(x) in related_down for x in anc[s]):
            continue
        ann = lookup.get(disease_id, onto.ids[s], drop_ref) or {}
        to_check.append({
            "term": onto.ids[s], "label": onto.labels[s], "freq": round(float(f), 3),
            "frequency": ann.get("frequency", ""), "score": float(f * ic[s]),
            "passage": disease_system_id(disease_id, onto.systems[s]) if onto.systems[s] else disease_overview_id(disease_id),
        })
    to_check.sort(key=lambda x: -x["score"])
    # Drop an item related (ancestor or descendant) to one already chosen: no near duplicates.
    chosen, kept = [], []
    for item in to_check:
        i = onto.index[item["term"]]
        if any(onto.is_ancestor(i, j) or onto.is_ancestor(j, i) for j in chosen):
            continue
        chosen.append(i)
        kept.append(item)
    to_check = kept
    n_support = sum(1 for m in matches if m["verdict"] == "supports")
    return {
        "disease_id": disease_id,
        "matches": matches,
        "contradictions": contradictions,
        "to_check": to_check[:6],
        "explained_share": round(n_support / max(len(matches), 1), 3),
        "profile_size": int(hi - lo),
    }
