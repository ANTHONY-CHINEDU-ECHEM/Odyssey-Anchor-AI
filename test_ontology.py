import numpy as np
import pandas as pd
import pytest

from odyssey_anchor.ontology.graph import Ontology


def test_ancestors_include_self_and_root(tiny):
    onto = tiny["onto"]
    focal = onto.index["HP:0007359"]
    anc = {onto.ids[i] for i in onto.ancestor_lists[focal]}
    assert {"HP:0007359", "HP:0001250", "HP:0000707", "HP:0000118", "HP:0000001"} == anc


def test_anc_matrix_matches_lists(tiny):
    onto = tiny["onto"]
    for i, a in enumerate(onto.ancestor_lists):
        row = set(onto.ANC[i].indices.tolist())
        assert row == set(a.tolist())


def test_most_specific_drops_ancestors(tiny):
    onto = tiny["onto"]
    idx = onto.idx(["HP:0001250", "HP:0007359", "HP:0001252"])
    kept = {onto.ids[i] for i in onto.most_specific(idx)}
    assert kept == {"HP:0007359", "HP:0001252"}


def test_phenotype_mask_excludes_inheritance(tiny):
    onto = tiny["onto"]
    assert onto.phenotype_mask[onto.index["HP:0001250"]]
    assert not onto.phenotype_mask[onto.index["HP:0000006"]]


def test_mica_is_shared_seizure(tiny):
    onto, ic = tiny["onto"], tiny["ic"]
    a = onto.mica(onto.index["HP:0007359"], onto.index["HP:0002069"], ic)
    assert onto.ids[a] == "HP:0001250"


def test_is_ancestor(tiny):
    onto = tiny["onto"]
    assert onto.is_ancestor(onto.index["HP:0001250"], onto.index["HP:0007359"])
    assert not onto.is_ancestor(onto.index["HP:0007359"], onto.index["HP:0001250"])


def test_cycle_detected():
    terms = pd.DataFrame({"id": ["HP:1", "HP:2"], "label": ["a", "b"], "synonyms": [[], []], "definition": ["", ""],
                          "system": ["", ""], "obsolete": [False, False]})
    edges = pd.DataFrame({"child": ["HP:1", "HP:2"], "parent": ["HP:2", "HP:1"]})
    onto = Ontology.from_tables(terms, edges)
    with pytest.raises(ValueError):
        _ = onto.ancestor_lists


def test_depth(tiny):
    onto = tiny["onto"]
    assert onto.depth[onto.index["HP:0000001"]] == 0
    assert onto.depth[onto.index["HP:0007359"]] == 4
    assert np.all(onto.depth >= 0)
