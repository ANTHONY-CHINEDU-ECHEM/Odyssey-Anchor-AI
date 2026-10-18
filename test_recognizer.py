import pytest

from odyssey_anchor.extract.recognizer import Recognizer
from odyssey_anchor.extract.render import render


@pytest.fixture(scope="module")
def rec(tiny):
    return Recognizer(tiny["onto"], tiny["ic"])


def statuses(rec, text):
    return {f.hpo_id: f.status for f in rec.extract(text)}


def test_present(rec):
    assert statuses(rec, "He has seizures and hypotonia.") == {"HP:0001250": "present", "HP:0001252": "present"}


def test_negation_scope_covers_coordinated_list(rec):
    s = statuses(rec, "No seizures, myopia or hypotonia.")
    assert set(s.values()) == {"absent"} and len(s) == 3


def test_negation_stops_at_but(rec):
    s = statuses(rec, "No seizures but marked hypotonia.")
    assert s == {"HP:0001250": "absent", "HP:0001252": "present"}


def test_post_negation(rec):
    assert statuses(rec, "Ectopia lentis was absent.") == {"HP:0001083": "absent"}


def test_family_history(rec):
    assert statuses(rec, "Her mother has myopia.") == {"HP:0000545": "family"}


def test_self_reference_resets_family(rec):
    s = statuses(rec, "Her mother is well and she has myopia.")
    assert s == {"HP:0000545": "present"}


def test_uncertain(rec):
    assert statuses(rec, "Possible ectopia lentis.") == {"HP:0001083": "uncertain"}


def test_synonym(rec):
    f = rec.extract("Low muscle tone noted.")
    assert f[0].hpo_id == "HP:0001252" and f[0].via == "exact synonym"


def test_spans_point_into_text(rec):
    text = "First line.\nNo myopia here. Seizures."
    for f in rec.extract(text):
        assert text[f.start:f.end].lower() == f.text.lower()


def test_to_terms(rec):
    t = rec.to_terms(rec.extract("Seizures. No myopia. Mother has hypotonia. Possible ectopia lentis."))
    assert t == {"observed": ["HP:0001250"], "excluded": ["HP:0000545"], "uncertain": ["HP:0001083"],
                 "family": ["HP:0001252"]}


def test_render_round_trip(tiny, rec):
    note = render(tiny["onto"], ["HP:0007359", "HP:0001252"], ["HP:0001083"], "FEMALE", 4.0, seed=3)
    out = rec.to_terms(rec.extract(note["text"]))
    assert "HP:0001083" in out["excluded"]
    assert "HP:0001083" not in out["observed"]
    assert render(tiny["onto"], ["HP:0007359"], [], "MALE", None, seed=3)["text"] == \
        render(tiny["onto"], ["HP:0007359"], [], "MALE", None, seed=3)["text"]
