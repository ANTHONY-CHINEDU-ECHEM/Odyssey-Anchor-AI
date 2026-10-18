"""Disease phenotype profiles as sparse matrices, with strict source removal.

For a set of diseases the profile holds four matrices over HPO terms:

  DIR  direct annotation frequency of each phenotype
  NOT  explicit NOT annotations (the disease is documented not to show it)
  F    propagated frequency: F[d, t] is the highest frequency of any annotation
       of d at t or below t, the chance that a patient with d shows something
       that a clinician would also record as t
  C    the binary support of F (closure of the annotations)

`without_reference` rebuilds the profile for the diseases touched by one
paper, dropping every annotation whose only references are that paper. The
evaluation applies it to every case so a patient is never matched against
annotations that were curated from the patient's own publication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import config
from .graph import Ontology


@dataclass
class Profiles:
    disease_ids: list[str]
    DIR: sp.csr_matrix
    NOT: sp.csr_matrix
    F: sp.csr_matrix
    C: sp.csr_matrix

    @property
    def n(self) -> int:
        return len(self.disease_ids)

    def csc(self):
        if not hasattr(self, "_csc"):
            self._csc = (self.F.tocsc(), self.C.tocsc(), self.DIR.tocsc(), self.NOT.tocsc())
        return self._csc

    @property
    def size(self) -> np.ndarray:
        return np.diff(self.DIR.indptr)


def _propagate(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, onto: Ontology, n_rows: int) -> sp.csr_matrix:
    """Max propagate (row, term, value) triples up to every ancestor of the term."""
    if len(rows) == 0:
        return sp.csr_matrix((n_rows, onto.n), dtype=np.float32)
    anc = onto.ancestor_lists
    lens = np.fromiter((len(anc[c]) for c in cols), dtype=np.int64, count=len(cols))
    r = np.repeat(rows, lens)
    v = np.repeat(vals, lens)
    c = np.concatenate([anc[x] for x in cols])
    key = r.astype(np.int64) * onto.n + c
    order = np.lexsort((-v, key))
    key, v = key[order], v[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    key, v = key[first], v[first]
    return sp.csr_matrix((v.astype(np.float32), (key // onto.n, key % onto.n)), shape=(n_rows, onto.n))


def build(annotations: pd.DataFrame, onto: Ontology, disease_ids: list[str] | None = None) -> Profiles:
    pheno = annotations[annotations.aspect.eq("P") & annotations.hpo_id.isin(onto.index)]
    if disease_ids is None:
        disease_ids = sorted(annotations.disease_id.unique())
    row_of = {d: i for i, d in enumerate(disease_ids)}
    pheno = pheno[pheno.disease_id.isin(row_of)]
    n = len(disease_ids)

    pos = pheno[~pheno.negated]
    rows = pos.disease_id.map(row_of).to_numpy(np.int64)
    cols = pos.hpo_id.map(onto.index).to_numpy(np.int64)
    vals = pos.freq.to_numpy(np.float32)
    # Duplicate (disease, term) rows: keep the highest frequency.
    if len(rows):
        key = rows * onto.n + cols
        order = np.lexsort((-vals, key))
        key, vals2 = key[order], vals[order]
        keep = np.ones(len(key), dtype=bool)
        keep[1:] = key[1:] != key[:-1]
        key, vals2 = key[keep], vals2[keep]
        rows, cols, vals = key // onto.n, key % onto.n, np.maximum(vals2, 1e-3)
    DIR = sp.csr_matrix((vals, (rows, cols)), shape=(n, onto.n), dtype=np.float32)

    neg = pheno[pheno.negated]
    NOT = sp.csr_matrix(
        (np.ones(len(neg), dtype=np.float32),
         (neg.disease_id.map(row_of).to_numpy(), neg.hpo_id.map(onto.index).to_numpy())),
        shape=(n, onto.n),
    )
    NOT.data[:] = 1.0
    NOT.sum_duplicates()
    NOT.data[:] = 1.0

    F = _propagate(rows, cols, vals, onto, n)
    C = F.copy()
    C.data[:] = 1.0
    return Profiles(disease_ids=list(disease_ids), DIR=DIR, NOT=NOT, F=F, C=C)


class StrictIndex:
    """Find, for a paper, the annotations that rest on that paper alone."""

    def __init__(self, annotations: pd.DataFrame):
        pheno = annotations[annotations.aspect.eq("P")]
        single = pheno[pheno.refs.apply(len).eq(1)]
        only = single.assign(ref=single.refs.str[0])
        only = only[only.ref.str.startswith("PMID:")]
        self.rows_by_ref = only.groupby("ref").apply(lambda g: g.index.to_numpy(), include_groups=False).to_dict()
        self.annotations = annotations

    def affected(self, pmid: str | None) -> tuple[list[str], pd.DataFrame | None]:
        """Diseases changed by removing the paper, and their remaining annotations."""
        if not pmid or pmid not in self.rows_by_ref:
            return [], None
        drop = self.rows_by_ref[pmid]
        diseases = sorted(self.annotations.loc[drop, "disease_id"].unique())
        remaining = self.annotations[self.annotations.disease_id.isin(diseases)].drop(index=drop)
        return diseases, remaining


def strict_profiles(strict: StrictIndex, pmid: str | None, onto: Ontology) -> Profiles | None:
    diseases, remaining = strict.affected(pmid)
    if not diseases:
        return None
    return build(remaining, onto, disease_ids=diseases)


def information_content(profiles: Profiles) -> np.ndarray:
    """Annotation based information content: IC(t) = -log(share of diseases whose closure holds t)."""
    counts = np.asarray(profiles.C.sum(axis=0)).ravel()
    n = profiles.n
    p = (counts + 1.0) / (n + 1.0)
    return (-np.log(p)).astype(np.float32)


def background(profiles: Profiles) -> np.ndarray:
    """Background probability of each term: mean propagated frequency over diseases, floored."""
    mean = np.asarray(profiles.F.sum(axis=0)).ravel() / profiles.n
    return np.clip(mean, 1.0 / (profiles.n * 4.0), 0.999).astype(np.float32)


def load_annotations() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "annotations.parquet")
