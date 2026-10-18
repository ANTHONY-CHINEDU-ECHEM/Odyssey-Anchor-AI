import numpy as np

from odyssey_anchor.rank.features import ranks
from odyssey_anchor.rank.scorers import LRParams, Patient, Scorer


def scorer(tiny):
    return Scorer(tiny["onto"], tiny["ic"], tiny["bg"], LRParams())


def row(tiny, d):
    return tiny["prof"].disease_ids.index(d)


def test_exact_match_beats_others(tiny):
    s = scorer(tiny)
    pat = Patient.from_terms(tiny["onto"], ["HP:0007359"])
    lr = s.likelihood_ratio(pat, tiny["prof"])["lr"]
    assert np.argmax(lr) == row(tiny, "OMIM:1")


def test_patient_general_term_matches_specific_annotation(tiny):
    s = scorer(tiny)
    pat = Patient.from_terms(tiny["onto"], ["HP:0001250"])
    lr = s.likelihood_ratio(pat, tiny["prof"])["lr"]
    # OMIM:1 shows seizures in 90%, OMIM:3 in 20%, OMIM:2 is documented NOT to.
    assert lr[row(tiny, "OMIM:1")] > lr[row(tiny, "OMIM:3")] > lr[row(tiny, "OMIM:2")]


def test_not_annotation_penalised(tiny):
    s = scorer(tiny)
    pat = Patient.from_terms(tiny["onto"], ["HP:0001250"])
    out = s.likelihood_ratio(pat, tiny["prof"])
    assert out["not_conflicts"][row(tiny, "OMIM:2")] == 1
    assert out["lr"][row(tiny, "OMIM:2")] < 0


def test_excluded_feature_lowers_disease_that_shows_it(tiny):
    s = scorer(tiny)
    base = Patient.from_terms(tiny["onto"], ["HP:0001252"])
    excl = Patient.from_terms(tiny["onto"], ["HP:0001252"], ["HP:0007359"])
    a = s.likelihood_ratio(base, tiny["prof"])["lr"]
    b = s.likelihood_ratio(excl, tiny["prof"])["lr"]
    assert b[row(tiny, "OMIM:1")] < a[row(tiny, "OMIM:1")]
    assert np.isclose(b[row(tiny, "OMIM:2")] - a[row(tiny, "OMIM:2")], b[row(tiny, "OMIM:3")] - a[row(tiny, "OMIM:3")], atol=0.3)


def test_excluded_ancestor_of_observed_is_dropped(tiny):
    pat = Patient.from_terms(tiny["onto"], ["HP:0007359"], ["HP:0001250"])
    assert len(pat.excluded) == 0


def test_resnik_bma_prefers_overlap(tiny):
    s = scorer(tiny)
    pat = Patient.from_terms(tiny["onto"], ["HP:0001083", "HP:0000545"])
    r = s.resnik(pat, tiny["prof"])
    assert np.argmax(r["bma"]) == row(tiny, "OMIM:2")
    assert np.all(r["bma"] >= 0)


def test_exact_share(tiny):
    s = scorer(tiny)
    pat = Patient.from_terms(tiny["onto"], ["HP:0001252", "HP:0001083"])
    e = s.exact_share(pat, tiny["prof"])
    assert e[row(tiny, "OMIM:1")] == 0.5 and e[row(tiny, "OMIM:2")] == 0.5


def test_competition_ranks():
    assert ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [1, 4, 1, 3]
