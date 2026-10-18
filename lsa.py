"""Latent semantic analysis: sublinear tf idf followed by a truncated SVD.

A dense semantic space that needs no downloaded embedding model, so the whole
system runs offline. Queries and new documents are projected with the fitted
vocabulary and components.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import config


def _identity(x):
    return x


class LSA:
    def __init__(self, dims: int = 192):
        self.dims = dims

    def fit(self, docs: list[list[str]]) -> "LSA":
        self.vec = TfidfVectorizer(analyzer=_identity, sublinear_tf=True, min_df=2, dtype=np.float32)
        X = self.vec.fit_transform(docs)
        dims = min(self.dims, X.shape[1] - 1, X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=dims, random_state=config.SEED, algorithm="randomized", n_iter=5)
        Z = self.svd.fit_transform(X)
        self.Z = self._normalise(Z)
        return self

    @staticmethod
    def _normalise(Z: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return (Z / np.maximum(n, 1e-9)).astype(np.float32)

    def embed(self, docs: list[list[str]]) -> np.ndarray:
        return self._normalise(self.svd.transform(self.vec.transform(docs)))

    def score(self, query: list[str]) -> np.ndarray:
        q = self.embed([query])[0]
        return self.Z @ q

    def score_new(self, query: list[str], docs: list[list[str]]) -> np.ndarray:
        q = self.embed([query])[0]
        return self.embed(docs) @ q
