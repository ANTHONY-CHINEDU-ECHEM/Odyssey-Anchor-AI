"""Stable identifiers for knowledge base passages.

These identifiers are fixed before the knowledge base is built so that the
frozen evaluation questions can name their gold passages in advance.
"""
from __future__ import annotations


def disease_overview_id(disease_id: str) -> str:
    return f"DIS|{disease_id}|overview"


def disease_system_id(disease_id: str, system_term: str) -> str:
    return f"DIS|{disease_id}|{system_term}"


def term_id(hpo_id: str) -> str:
    return f"HPO|{hpo_id}"


def gene_id(symbol: str) -> str:
    return f"GENE|{symbol}"


def group_id(mondo_id: str) -> str:
    return f"GRP|{mondo_id}"


def kind_of(passage_id: str) -> str:
    return passage_id.split("|", 1)[0]
