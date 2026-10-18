import numpy as np
import pandas as pd
import pytest

from odyssey_anchor.rank.features import BROWSER_FEATURES, FEATURES, in_browser_shortlist, mid_rank, rounded
from odyssey_anchor.rank.fusion import LambdaMART, LinearRanker, TreeEnsemble, export_trees, fit_temperature


def toy_frame(n_cases=60, per=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cases):
        gold = rng.integers(per)
        for j in range(per):
            x = rng.normal(size=len(FEATURES))
            if j == gold:
                x[FEATURES.index("lr")] += 3.0
                x[FEATURES.index("bma")] += 1.5
            rows.append({"case_id": f"c{c}", "label": int(j == gold), **dict(zip(FEATURES, x))})
    return pd.DataFrame(rows)


def gold_rank(df, s):
    d = df.assign(s=s)
    return d.groupby("case_id").apply(lambda g: (g.s > g.s[g.label == 1].iloc[0]).sum() + 1, include_groups=False)


def test_linear_ranker_learns_signal_and_round_trips():
    df = toy_frame()
    m = LinearRanker().fit(df)
    assert gold_rank(df, m.score(df)).mean() < 2.0
    m2 = LinearRanker.from_json(m.to_json())
    assert np.allclose(m.score(df), m2.score(df))


def test_exported_trees_score_like_lightgbm():
    df = toy_frame()
    m = LambdaMART(features=BROWSER_FEATURES).fit(df)
    te = TreeEnsemble({"features": BROWSER_FEATURES, "trees": export_trees(m.model.booster_)})
    assert np.allclose(te.score(df.head(80)), m.score(df.head(80)), atol=1e-6)


def test_temperature_is_positive_and_finite():
    df = toy_frame()
    T = fit_temperature(df, LinearRanker().fit(df).score(df))
    assert 0.05 <= T <= 20


def test_mid_rank_splits_ties():
    s = np.array([1.0, 1.0, 1.0, 0.5])
    assert mid_rank(s, 0) == 2.0
    assert mid_rank(s, 3) == 4.0


def test_rounding_makes_near_ties_exact():
    a = np.array([2.4668272, 2.4668271], dtype=np.float32)
    r = rounded(a)
    assert r[0] == r[1]


def test_browser_shortlist_mask():
    df = pd.DataFrame({"logrank_lr": np.log([1, 200, 500]), "logrank_bma": np.log([900, 50, 900]),
                       "logrank_d2p": np.log([900, 900, 900])})
    assert in_browser_shortlist(df).tolist() == [True, True, False]


def test_constrained_linear_weights_respect_directions():
    from odyssey_anchor.rank.features import MONOTONE
    df = toy_frame()
    # Plant the curation artifact: the true disease has more excluded features it usually shows.
    df.loc[df.label.eq(1), "exc_against"] += 2.0
    m = LinearRanker().fit(df)
    for f, w in zip(m.features, m.w):
        assert MONOTONE[f] == 0 or np.sign(w) in (0, MONOTONE[f]), f
    assert LinearRanker(constrained=False).fit(df).w[FEATURES.index("exc_against")] > 0


def test_constrained_lambdamart_never_rewards_contradictions():
    df = toy_frame()
    df.loc[df.label.eq(1), "exc_against"] += 2.0
    m = LambdaMART().fit(df)
    base = df.head(50).copy()
    more = base.assign(exc_against=base.exc_against + 3.0)
    assert np.all(m.score(more) <= m.score(base) + 1e-9)
    free = LambdaMART(constrained=False).fit(df)
    assert np.any(free.score(more) > free.score(base) + 1e-6)
