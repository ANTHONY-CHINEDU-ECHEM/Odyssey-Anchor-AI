"""The HPO graph as integer indexed sparse matrices.

ANC is a square boolean matrix with ANC[i, j] = 1 when term j is term i or one
of its ancestors. Everything else (closure of a patient, most informative
common ancestor, propagation of disease annotations) is built from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import config


@dataclass
class Ontology:
    ids: list[str]
    labels: list[str]
    parents: list[list[int]]
    synonyms: list[list[str]]
    definitions: list[str]
    systems: list[str]
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.ids)}

    # construction --------------------------------------------------------
    @classmethod
    def from_tables(cls, terms: pd.DataFrame, edges: pd.DataFrame) -> "Ontology":
        terms = terms[~terms.obsolete].sort_values("id").reset_index(drop=True)
        ids = terms.id.tolist()
        index = {t: i for i, t in enumerate(ids)}
        parents: list[list[int]] = [[] for _ in ids]
        for c, p in zip(edges.child, edges.parent):
            if c in index and p in index:
                parents[index[c]].append(index[p])
        return cls(
            ids=ids,
            labels=terms.label.tolist(),
            parents=parents,
            synonyms=[list(s) for s in terms.synonyms],
            definitions=terms.definition.fillna("").tolist(),
            systems=terms.system.fillna("").tolist(),
        )

    @classmethod
    def load(cls) -> "Ontology":
        terms = pd.read_parquet(config.PROCESSED / "hpo_terms.parquet")
        edges = pd.read_parquet(config.PROCESSED / "hpo_edges.parquet")
        return cls.from_tables(terms, edges)

    # structure -------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.ids)

    @cached_property
    def ancestor_lists(self) -> list[np.ndarray]:
        """Ancestors of each term including the term itself, in topological memo order."""
        memo: list[np.ndarray | None] = [None] * self.n
        order = self._topological()
        for i in order:
            acc = {i}
            for p in self.parents[i]:
                acc.update(memo[p].tolist())  # parents precede children in the order
            memo[i] = np.fromiter(sorted(acc), dtype=np.int32)
        return memo  # type: ignore[return-value]

    def _topological(self) -> list[int]:
        children: list[list[int]] = [[] for _ in range(self.n)]
        indeg = np.zeros(self.n, dtype=np.int32)
        for c, ps in enumerate(self.parents):
            for p in ps:
                children[p].append(c)
                indeg[c] += 1
        queue = [i for i in range(self.n) if indeg[i] == 0]
        order = []
        while queue:
            i = queue.pop()
            order.append(i)
            for c in children[i]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != self.n:
            raise ValueError("HPO graph contains a cycle")
        return order

    @cached_property
    def children(self) -> list[list[int]]:
        ch: list[list[int]] = [[] for _ in range(self.n)]
        for c, ps in enumerate(self.parents):
            for p in ps:
                ch[p].append(c)
        return ch

    @cached_property
    def ANC(self) -> sp.csr_matrix:
        rows = np.concatenate([np.full(len(a), i, dtype=np.int32) for i, a in enumerate(self.ancestor_lists)])
        cols = np.concatenate(self.ancestor_lists)
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    @cached_property
    def ANC_csc(self) -> sp.csc_matrix:
        return self.ANC.tocsc()

    @cached_property
    def phenotype_mask(self) -> np.ndarray:
        root = self.index[config.PHENOTYPIC_ABNORMALITY]
        return np.asarray(self.ANC[:, root].todense()).ravel() > 0

    @cached_property
    def depth(self) -> np.ndarray:
        d = np.zeros(self.n, dtype=np.int32)
        for i in self._topological():
            d[i] = 0 if not self.parents[i] else 1 + max(d[p] for p in self.parents[i])
        return d

    # helpers ---------------------------------------------------------------
    def idx(self, terms) -> np.ndarray:
        return np.array([self.index[t] for t in terms if t in self.index], dtype=np.int32)

    def closure(self, idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate([self.ancestor_lists[i] for i in idx]))

    def is_ancestor(self, a: int, t: int) -> bool:
        """True when a is t or an ancestor of t."""
        anc = self.ancestor_lists[t]
        pos = np.searchsorted(anc, a)
        return pos < len(anc) and anc[pos] == a

    def most_specific(self, idx: np.ndarray) -> np.ndarray:
        """Drop any term that is an ancestor of another term in the set."""
        idx = np.unique(idx)
        keep = []
        s = set(idx.tolist())
        redundant = set()
        for i in idx:
            for a in self.ancestor_lists[i]:
                if a != i and a in s:
                    redundant.add(int(a))
        for i in idx:
            if int(i) not in redundant:
                keep.append(int(i))
        return np.array(keep, dtype=np.int32)

    def mica(self, a: int, b: int, ic: np.ndarray) -> int:
        common = np.intersect1d(self.ancestor_lists[a], self.ancestor_lists[b], assume_unique=True)
        # Highest information content; ties go to the deeper, more specific term.
        return int(common[np.lexsort((self.depth[common], ic[common]))[-1]])

    def label(self, i: int) -> str:
        return self.labels[i]
