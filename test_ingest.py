import pytest

from odyssey_anchor.ingest.hpoa import parse_frequency
from odyssey_anchor.ingest.mondo import normalise_xref, strip_treatment
from odyssey_anchor.ingest.phenopackets import age_years, parse_one


def test_frequency_forms():
    assert parse_frequency("HP:0040281") == pytest.approx(0.895)
    assert parse_frequency("1/1") == pytest.approx(0.75)        # smoothed, not certainty
    assert parse_frequency("7/12") == pytest.approx(7.5 / 13)
    assert parse_frequency("45%") == pytest.approx(0.45)
    assert parse_frequency("") == 0.5
    assert parse_frequency("HP:0040285") == 0.0


def test_treatment_sentences_removed():
    text = "A connective tissue disorder. Treatment is based on symptoms. It is caused by FBN1 variants."
    out, n = strip_treatment(text)
    assert n == 1 and "Treatment" not in out and "FBN1" in out


def test_xref_normalisation():
    assert normalise_xref("Orphanet:558") == "ORPHA:558"
    assert normalise_xref("OMIM:154700") == "OMIM:154700"
    assert normalise_xref("OMIM:PS154700") is None
    assert normalise_xref("DOID:14323") is None


def test_age_parsing():
    assert age_years({"age": {"iso8601duration": "P13Y"}}) == 13
    assert age_years({"age": {"iso8601duration": "P1Y6M"}}) == 1.5
    assert age_years({"gestationalAge": {"weeks": 30}}) == 0.0
    assert age_years(None) is None


def test_phenopacket_parse():
    doc = {"id": "c1", "subject": {"sex": "FEMALE"},
           "phenotypicFeatures": [{"type": {"id": "HP:1"}}, {"type": {"id": "HP:2"}, "excluded": True},
                                  {"type": {"id": "HP:1"}}],
           "interpretations": [{"diagnosis": {"disease": {"id": "OMIM:1", "label": "D"},
                                              "genomicInterpretations": [{"variantInterpretation": {"variationDescriptor": {"geneContext": {"symbol": "G1"}}}}]}}],
           "metaData": {"externalReferences": [{"id": "PMID:5", "description": "t"}]}}
    row = parse_one(doc, "G1", resolve=lambda t: t)
    assert row["observed"] == ["HP:1"] and row["excluded"] == ["HP:2"]
    assert row["genes"] == ["G1"] and row["pmid"] == "PMID:5" and row["sex"] == "FEMALE"
    assert parse_one({"id": "x", "interpretations": []}, "c", None) is None
