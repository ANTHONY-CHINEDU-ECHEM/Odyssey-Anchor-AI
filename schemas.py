"""Request bodies for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyseIn(BaseModel):
    note: str | None = Field(None, description="Free text clinical description; identifiers are redacted before use")
    observed: list[str] = Field(default_factory=list, description="HPO identifiers recorded as present")
    excluded: list[str] = Field(default_factory=list, description="HPO identifiers recorded as absent")
    question: str | None = Field(None, description="Optional knowledge question answered from cited passages")
    sex: str = Field("UNKNOWN_SEX", description="MALE, FEMALE or UNKNOWN_SEX")
    age_years: float | None = Field(None, ge=0, le=120)
    strict_pmid: str | None = Field(None, description="For published cases: remove annotations resting only on this paper")


class AskIn(BaseModel):
    question: str = Field(..., min_length=3)
    context: list[str] = Field(default_factory=list, description="Disease identifiers in the current differential")


class ExtractIn(BaseModel):
    note: str = Field(..., min_length=3)
