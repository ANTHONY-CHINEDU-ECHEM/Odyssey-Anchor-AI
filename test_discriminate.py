import numpy as np

from odyssey_anchor.reason.discriminate import entropy
from odyssey_anchor.reason.genes import GeneIndex
import pandas as pd


def test_entropy_bounds():
    assert np.isclose(entropy(np.array([0.5, 0.5])), 1.0)
    assert np.isclose(entropy(np.array([1.0, 0.0])), 0.0, atol=1e-6)


def test_information_gain_prefers_discriminating_question():
    pi = np.array([0.5, 0.5])
    # Question A: shown by disease 1 only. Question B: shown by both.
    like = np.array([[0.95, 0.02], [0.9, 0.9]])
    p_yes = like @ pi
    post_yes = like * pi / p_yes[:, None]
    post_no = (1 - like) * pi / (1 - p_yes)[:, None]
    eig = entropy(pi) - (p_yes * entropy(post_yes) + (1 - p_yes) * entropy(post_no))
    assert eig[0] > 0.7 and eig[1] < 0.01


def test_gene_panel_weights_by_posterior():
    dg = pd.DataFrame({"disease_id": ["D1", "D2", "D2", "D3"], "gene_symbol": ["A", "A", "B", "C"]})
    gi = GeneIndex(dg)
    panel = gi.panel(["D1", "D2", "D3"], [0.5, 0.3, 0.2], {"D1": "one", "D2": "two", "D3": "three"})
    assert [p["gene"] for p in panel] == ["A", "B", "C"]
    assert np.isclose(panel[0]["weight"], 0.8)
    assert gi.gene_scores(["D1", "D3"], [2.0, 5.0]) == {"A": 2.0, "C": 5.0}
