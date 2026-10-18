"""Okapi BM25 over a sparse document term matrix.

Weights are precomputed so that scoring a query is one sparse product. New
documents (used when the strict protocol rewrites a disease) are scored with
the same idf and average length as the fitted collection.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}

    def fit(self, docs: list[list[str]]) -> "BM25":
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, doc in enumerate(docs):
            counts = Counter(doc)
            lengths[i] = len(doc)
            for tok, c in counts.items():
                j = vocab.setdefault(tok, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), len(vocab)))
        df = np.bincount(tf.indices, minlength=len(vocab)).astype(np.float32)
        n = len(docs)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.avgdl = float(lengths.mean()) if n else 1.0
        self.W = self._weights(tf, lengths)
        return self

    def _weights(self, tf: sp.csr_matrix, lengths: np.ndarray) -> sp.csr_matrix:
        tf = tf.tocsr().astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * lengths / self.avgdl)
        row_norm = np.repeat(norm, np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + row_norm)
        W = sp.csr_matrix((data * self.idf[tf.indices], tf.indices, tf.indptr), shape=tf.shape)
        return W

    def query_vector(self, query: list[str]) -> sp.csr_matrix:
        idx = sorted({self.vocab[t] for t in query if t in self.vocab})
        return sp.csr_matrix((np.ones(len(idx), dtype=np.float32), (np.zeros(len(idx), int), idx)),
                             shape=(1, len(self.vocab)))

    def score(self, query: list[str]) -> np.ndarray:
        q = self.query_vector(query)
        return np.asarray((self.W @ q.T).todense()).ravel()

    def score_new(self, query: list[str], docs: list[list[str]]) -> np.ndarray:
        """Score documents outside the fitted collection with the fitted statistics."""
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, doc in enumerate(docs):
            lengths[i] = len(doc)
            for tok, c in Counter(doc).items():
                if tok in self.vocab:
                    rows.append(i)
                    cols.append(self.vocab[tok])
                    vals.append(c)
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), len(self.vocab)))
        W = self._weights(tf, lengths)
        q = self.query_vector(query)
        return np.asarray((W @ q.T).todense()).ravel()
