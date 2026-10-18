import numpy as np

from odyssey_anchor.ontology import profiles as P


def col(prof, onto, did, tid):
    return float(prof.F[prof.disease_ids.index(did), onto.index[tid]])


def test_propagation_takes_maximum(tiny):
    onto, prof = tiny["onto"], tiny["prof"]
    # OMIM:1 annotated focal seizure 0.9: seizure and nervous system inherit 0.9
    assert col(prof, onto, "OMIM:1", "HP:0001250") == np.float32(0.9)
    assert col(prof, onto, "OMIM:1", "HP:0000707") == np.float32(0.9)
    # OMIM:3: hypotonia 0.9 and tonic clonic 0.2 both under nervous system: max is 0.9
    assert col(prof, onto, "OMIM:3", "HP:0000707") == np.float32(0.9)
    assert col(prof, onto, "OMIM:3", "HP:0001250") == np.float32(0.2)


def test_not_annotations_kept_apart(tiny):
    onto, prof = tiny["onto"], tiny["prof"]
    r = prof.disease_ids.index("OMIM:2")
    assert prof.NOT[r, onto.index["HP:0001250"]] == 1
    assert prof.F[r, onto.index["HP:0001250"]] == 0


def test_information_content_grows_with_depth(tiny):
    onto, ic = tiny["onto"], tiny["ic"]
    assert ic[onto.index["HP:0007359"]] >= ic[onto.index["HP:0001250"]] >= ic[onto.index["HP:0000707"]]
    assert ic[onto.index["HP:0000118"]] < 0.5


def test_strict_removes_single_paper_annotations(tiny):
    onto, ann = tiny["onto"], tiny["ann"]
    strict = P.StrictIndex(ann)
    diseases, remaining = strict.affected("PMID:333")
    assert diseases == ["OMIM:3"]
    assert remaining[remaining.disease_id.eq("OMIM:3")].empty
    # PMID:222 backs ectopia lentis alone, but myopia also cites OMIM:2 and must survive.
    sp = P.strict_profiles(strict, "PMID:222", onto)
    r = sp.disease_ids.index("OMIM:2")
    assert sp.DIR[r, onto.index["HP:0001083"]] == 0
    assert sp.DIR[r, onto.index["HP:0000545"]] > 0


def test_strict_noop_for_unknown_paper(tiny):
    strict = P.StrictIndex(tiny["ann"])
    assert strict.affected("PMID:999") == ([], None)
    assert P.strict_profiles(strict, None, tiny["onto"]) is None


def test_background_in_range(tiny):
    bg = tiny["bg"]
    assert np.all(bg > 0) and np.all(bg < 1)
