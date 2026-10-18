"""Gene panel suggestion: the genes behind the leading candidates, weighted by posterior.

The panel is a prioritised list for a clinician choosing what to test or which
variants to review first. It never states that a gene is causative.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd


class GeneIndex:
    def __init__(self, disease_genes: pd.DataFrame):
        self.by_disease = disease_genes.groupby("disease_id").gene_symbol.apply(lambda s: sorted(set(s))).to_dict()
        self.by_gene = disease_genes.groupby("gene_symbol").disease_id.apply(lambda s: sorted(set(s))).to_dict()

    def panel(self, candidates: list[str], posterior: list[float], names: dict[str, str], k: int = 10) -> list[dict]:
        score: dict[str, float] = defaultdict(float)
        via: dict[str, list[str]] = defaultdict(list)
        for d, p in zip(candidates, posterior):
            for g in self.by_disease.get(d, []):
                score[g] += float(p)
                via[g].append(d)
        ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [{"gene": g, "weight": round(s, 4), "diseases": via[g],
                 "disease_names": [names.get(d, d) for d in via[g]]} for g, s in ranked]

    def panel_concepts(self, concepts: list[tuple[list[str], float]], names: dict[str, str], k: int = 10) -> list[dict]:
        """Genes of every record of each concept, weighted by the concept probability (counted once per concept)."""
        score: dict[str, float] = defaultdict(float)
        via: dict[str, list[str]] = defaultdict(list)
        for members, p in concepts:
            genes = sorted({g for d in members for g in self.by_disease.get(d, [])})
            for g in genes:
                score[g] += float(p)
                via[g].append(next(d for d in members if g in self.by_disease.get(d, [])))
        ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [{"gene": g, "weight": round(s, 4), "diseases": via[g],
                 "disease_names": [names.get(d, d) for d in via[g]]} for g, s in ranked]

    def gene_scores(self, disease_ids: list[str], scores) -> dict[str, float]:
        """Gene ranking for evaluation: a gene scores as its best disease."""
        best: dict[str, float] = {}
        for d, s in zip(disease_ids, scores):
            for g in self.by_disease.get(d, []):
                if g not in best or s > best[g]:
                    best[g] = float(s)
        return best
