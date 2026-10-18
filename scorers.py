"""Phenotype scorers over every disease at once.

Patient
  observed: the most specific observed phenotype terms
  excluded: phenotype terms the clinician recorded as absent

Semantic similarity (Resnik best match average)
  p2d[d]  mean over patient terms t of the information content of the most
          informative ancestor shared by t and any annotation of d
  d2p[d]  frequency weighted mean over annotations s of d of the information
          content of the most informative ancestor shared by s and the patient
  bma[d]  the average of the two directions

Likelihood ratio model
  For each observed term t the chance that a patient with d shows t is the
  largest of three routes, each a probability not a score:
    exact or more specific annotation   F[d, t]
    a more general annotation a         DIR[d, a] * p(t) / p(a)
    a shared ancestor a                 lambda * F[d, a] * p(t) / p(a)
  smoothed toward the background p(t) and divided by it. A NOT annotation that
  covers t multiplies the ratio by a floor. For each excluded term e the ratio
  is (1 - 0.95 F[d, e]) / (1 - p(e)), floored. The disease score is the sum of
  log ratios, which is a log posterior odds under a uniform prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..ontology.graph import Ontology
from ..ontology.profiles import Profiles


@dataclass
class Patient:
    observed: np.ndarray            # most specific observed term indices
    excluded: np.ndarray            # excluded term indices
    sex: str = "UNKNOWN_SEX"
    age_years: float | None = None
    closure: np.ndarray = field(default=None)  # type: ignore[assignment]

    @classmethod
    def from_terms(cls, onto: Ontology, observed, excluded=(), sex="UNKNOWN_SEX", age_years=None) -> "Patient":
        obs = onto.idx(observed)
        obs = obs[onto.phenotype_mask[obs]] if len(obs) else obs
        obs = onto.most_specific(obs)
        exc = onto.idx(excluded)
        exc = exc[onto.phenotype_mask[exc]] if len(exc) else exc
        obs_closure = set(onto.closure(obs).tolist())
        # An excluded term that is an ancestor of an observed one contradicts the record; drop it.
        exc = np.array(sorted({int(e) for e in exc if int(e) not in obs_closure}), dtype=np.int32)
        p = cls(observed=obs, excluded=exc, sex=sex or "UNKNOWN_SEX", age_years=age_years)
        p.closure = onto.closure(obs)
        return p


@dataclass
class LRParams:
    lam: float = 0.1       # weight of a match through a shared ancestor
    eta: float = 0.02      # smoothing toward the background
    not_floor: float = config.EXCLUDED_PENALTY_FLOOR
    exc_floor: float = config.EXCLUDED_PENALTY_FLOOR


class Scorer:
    def __init__(self, onto: Ontology, ic: np.ndarray, bg: np.ndarray, params: LRParams | None = None):
        self.onto = onto
        self.ic = ic
        self.bg = bg
        self.params = params or LRParams()

    # semantic similarity ------------------------------------------------------
    def resnik(self, pat: Patient, prof: Profiles) -> dict[str, np.ndarray]:
        n = prof.n
        F_c, C_c, DIR_c, _ = prof.csc()
        ic = self.ic
        per_term = np.zeros((len(pat.observed), n), dtype=np.float32)
        for k, t in enumerate(pat.observed):
            best = per_term[k]
            for a in self.onto.ancestor_lists[t]:
                lo, hi = C_c.indptr[a], C_c.indptr[a + 1]
                if lo == hi:
                    continue
                rows = C_c.indices[lo:hi]
                best[rows] = np.maximum(best[rows], ic[a])
        p2d = per_term.mean(axis=0) if len(pat.observed) else np.zeros(n, dtype=np.float32)

        # f[s]: best information content shared between term s and the patient closure.
        if len(pat.closure):
            sub = self.onto.ANC_csc[:, pat.closure].tocsr()
            sub = sub.multiply(ic[pat.closure][None, :]).tocsr()
            f = np.asarray(sub.max(axis=1).todense()).ravel().astype(np.float32)
        else:
            f = np.zeros(self.onto.n, dtype=np.float32)
        weights = np.asarray(prof.DIR.sum(axis=1)).ravel()
        d2p = np.asarray(prof.DIR @ f).ravel() / np.maximum(weights, 1e-6)
        return {"p2d": p2d, "d2p": d2p.astype(np.float32), "bma": (0.5 * (p2d + d2p)).astype(np.float32),
                "per_term_ic": per_term}

    # likelihood ratio -----------------------------------------------------------
    def term_likelihood(self, t: int, prof: Profiles) -> np.ndarray:
        """P(patient shows t | d) for every disease, before smoothing."""
        F_c, _, DIR_c, _ = prof.csc()
        bg, lam = self.bg, self.params.lam
        out = np.zeros(prof.n, dtype=np.float32)
        lo, hi = F_c.indptr[t], F_c.indptr[t + 1]
        out[F_c.indices[lo:hi]] = F_c.data[lo:hi]
        for a in self.onto.ancestor_lists[t]:
            if a == t:
                continue
            ratio = bg[t] / bg[a]
            lo, hi = DIR_c.indptr[a], DIR_c.indptr[a + 1]
            if lo < hi:
                rows = DIR_c.indices[lo:hi]
                out[rows] = np.maximum(out[rows], DIR_c.data[lo:hi] * ratio)
            lo, hi = F_c.indptr[a], F_c.indptr[a + 1]
            if lo < hi:
                rows = F_c.indices[lo:hi]
                out[rows] = np.maximum(out[rows], lam * F_c.data[lo:hi] * ratio)
        return out

    def not_conflict(self, t: int, prof: Profiles) -> np.ndarray:
        _, _, _, NOT_c = prof.csc()
        hit = np.zeros(prof.n, dtype=bool)
        for a in self.onto.ancestor_lists[t]:
            lo, hi = NOT_c.indptr[a], NOT_c.indptr[a + 1]
            if lo < hi:
                hit[NOT_c.indices[lo:hi]] = True
        return hit

    def likelihood_ratio(self, pat: Patient, prof: Profiles) -> dict[str, np.ndarray]:
        n = prof.n
        eta = self.params.eta
        obs_log = np.zeros((len(pat.observed), n), dtype=np.float32)
        conflicts = np.zeros(n, dtype=np.float32)
        for k, t in enumerate(pat.observed):
            p_t = self.bg[t]
            like = (1 - eta) * self.term_likelihood(t, prof) + eta * p_t
            lr = like / p_t
            hit = self.not_conflict(t, prof)
            if hit.any():
                lr[hit] = np.minimum(lr[hit], 1.0) * self.params.not_floor
                conflicts += hit
            obs_log[k] = np.log(lr)
        F_c, _, _, _ = prof.csc()
        exc_log = np.zeros((len(pat.excluded), n), dtype=np.float32)
        exc_against = np.zeros(n, dtype=np.float32)
        for k, e in enumerate(pat.excluded):
            col = np.zeros(n, dtype=np.float32)
            lo, hi = F_c.indptr[e], F_c.indptr[e + 1]
            col[F_c.indices[lo:hi]] = F_c.data[lo:hi]
            lr = np.maximum((1.0 - 0.95 * col) / (1.0 - min(self.bg[e], 0.95)), self.params.exc_floor)
            exc_log[k] = np.log(lr)
            exc_against += (col >= 0.5)
        total = obs_log.sum(axis=0) + exc_log.sum(axis=0)
        return {
            "lr": total,
            "lr_obs": obs_log.sum(axis=0),
            "lr_exc": exc_log.sum(axis=0),
            "support": (obs_log > np.log(2.0)).sum(axis=0).astype(np.float32),
            "against": (obs_log < np.log(0.5)).sum(axis=0).astype(np.float32),
            "not_conflicts": conflicts,
            "exc_against": exc_against,
            "obs_log": obs_log,
        }

    def exact_share(self, pat: Patient, prof: Profiles) -> np.ndarray:
        """Share of observed terms annotated to the disease at that term or below."""
        if len(pat.observed) == 0:
            return np.zeros(prof.n, dtype=np.float32)
        F_c, _, _, _ = prof.csc()
        hits = np.zeros(prof.n, dtype=np.float32)
        for t in pat.observed:
            lo, hi = F_c.indptr[t], F_c.indptr[t + 1]
            hits[F_c.indices[lo:hi]] += 1
        return hits / len(pat.observed)
