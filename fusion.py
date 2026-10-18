"""The fused ranker: learn how to weigh the base rankers, on development cases only.

Two learners are compared with five fold cross validation grouped by cohort:

  linear pairwise   logistic regression on feature differences between the true
                    disease and shortlisted alternatives (RankNet with a linear
                    scorer). Interpretable, and portable to the browser.
  LambdaMART        LightGBM lambdarank over the same features.

Pre-registered choice: the linear model is the product unless LambdaMART beats
it by more than 0.02 in mean reciprocal rank on cross validation.

Posterior probabilities come from a softmax over the shortlist with a
temperature fitted on out of fold scores. Scores are aggregated to Mondo
concepts (OMIM and Orphanet records of one disease) by taking the maximum.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression

from .. import config
from .features import BROWSER_FEATURES, FEATURES, MONOTONE, in_browser_shortlist

MARGIN = 0.02
NEG_HARD, NEG_RANDOM = 25, 25


class LinearRanker:
    def __init__(self, C: float = 0.5, features: list[str] | None = None, constrained: bool = True):
        self.C = C
        self.features = list(features or FEATURES)
        self.constrained = constrained

    def fit(self, df: pd.DataFrame) -> "LinearRanker":
        X = df[self.features].to_numpy(np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Z = (X - self.mean) / self.std
        rng = np.random.default_rng(config.SEED)
        diffs = []
        case = df.case_id.to_numpy()
        label = df.label.to_numpy()
        lr = df.lr.to_numpy()
        starts = np.r_[0, np.flatnonzero(case[1:] != case[:-1]) + 1, len(case)]
        for a, b in zip(starts[:-1], starts[1:]):
            lab = label[a:b]
            if lab.sum() != 1:
                continue
            g = a + int(np.argmax(lab))
            neg = np.arange(a, b)[lab == 0]
            if len(neg) == 0:
                continue
            hard = neg[np.argsort(-lr[neg])[:NEG_HARD]]
            rest = np.setdiff1d(neg, hard)
            rand = rng.choice(rest, size=min(NEG_RANDOM, len(rest)), replace=False) if len(rest) else rest
            for n in np.concatenate([hard, rand]):
                diffs.append(Z[g] - Z[n])
        D = np.array(diffs)
        if not self.constrained:
            Xp = np.vstack([D, -D])
            yp = np.r_[np.ones(len(D)), np.zeros(len(D))]
            self.model = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
            self.model.fit(Xp, yp)
            self.w = self.model.coef_.ravel()
            return self
        # Pairwise logistic loss with each weight's sign bounded by MONOTONE (features are standardised,
        # so a weight's sign is the direction the feature pushes the score).
        lam = 1.0 / (self.C * len(D))

        def loss(w):
            z = D @ w
            l = np.logaddexp(0.0, -z).mean() + 0.5 * lam * w @ w
            g = -(D * (1.0 / (1.0 + np.exp(z)))[:, None]).mean(axis=0) + lam * w
            return l, g

        bounds = [(0, None) if MONOTONE.get(f, 0) > 0 else ((None, 0) if MONOTONE.get(f, 0) < 0 else (None, None))
                  for f in self.features]
        res = minimize(loss, np.zeros(D.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        self.w = res.x
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        Z = (df[self.features].to_numpy(np.float64) - self.mean) / self.std
        return Z @ self.w

    def to_json(self) -> dict:
        return {"kind": "linear_pairwise", "features": self.features, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "weights": self.w.tolist(), "C": self.C}

    @classmethod
    def from_json(cls, d: dict) -> "LinearRanker":
        m = cls(C=d.get("C", 0.5), features=d["features"])
        m.mean, m.std, m.w = np.array(d["mean"]), np.array(d["std"]), np.array(d["weights"])
        return m


class LambdaMART:
    def __init__(self, features: list[str] | None = None, constrained: bool = True):
        self.features = list(features or FEATURES)
        self.constrained = constrained

    def fit(self, df: pd.DataFrame) -> "LambdaMART":
        sizes = df.groupby("case_id", sort=False).size().to_numpy()
        self.model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            lambdarank_truncation_level=20, random_state=config.SEED, verbose=-1, n_jobs=2,
            monotone_constraints=[MONOTONE.get(f, 0) for f in self.features] if self.constrained else None,
            monotone_constraints_method="advanced",
        )
        self.model.fit(df[self.features].to_numpy(np.float64), df.label.to_numpy(), group=sizes)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].to_numpy(np.float64))


def attach_concepts(df: pd.DataFrame, concept_of: dict[str, str]) -> pd.DataFrame:
    def concept(d: str) -> str:
        c = concept_of.get(d)
        return c if isinstance(c, str) and c else d
    return df.assign(concept=df.disease_id.map(concept))


def case_ranks(df: pd.DataFrame, scores: np.ndarray, base: pd.DataFrame) -> pd.DataFrame:
    """Rank of the true disease per case, by identifier and by Mondo concept."""
    d = df[["case_id", "disease_id", "concept", "label"]].copy()
    d["score"] = scores
    out = []
    for cid, g in d.groupby("case_id", sort=False):
        b = base.loc[cid]
        if not b.in_shortlist:
            out.append((cid, b.shortlist_size + int(b.rank_lr), b.shortlist_size + int(b.rank_lr), False))
            continue
        gold = g[g.label.eq(1)].iloc[0]
        id_rank = int((g.score > gold.score).sum()) + 1
        conc = g.groupby("concept").score.max()
        c_rank = int((conc > conc[gold.concept]).sum()) + 1
        out.append((cid, id_rank, c_rank, True))
    return pd.DataFrame(out, columns=["case_id", "rank_id", "rank_concept", "in_shortlist"]).set_index("case_id")


def summarise(ranks: pd.Series) -> dict:
    r = ranks.to_numpy()
    res = {f"top{k}": float((r <= k).mean()) for k in config.TOP_K}
    res["mrr"] = float((1.0 / r).mean())
    res["median_rank"] = float(np.median(r))
    res["n"] = int(len(r))
    return res


def softmax_nll(T: float, groups: list[tuple[np.ndarray, int]]) -> float:
    total = 0.0
    for s, g in groups:
        z = s / T
        z = z - z.max()
        total += -(z[g] - np.log(np.exp(z).sum()))
    return total / max(len(groups), 1)


def fit_temperature(df: pd.DataFrame, scores: np.ndarray) -> float:
    groups = []
    d = df.assign(score=scores)
    for _, g in d.groupby("case_id", sort=False):
        if g.label.sum() == 1:
            groups.append((g.score.to_numpy(), int(np.argmax(g.label.to_numpy()))))
    res = minimize_scalar(lambda T: softmax_nll(T, groups), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def cross_validate(F: pd.DataFrame, base: pd.DataFrame, folds: pd.Series) -> dict:
    F = F.assign(fold=F.case_id.map(folds))
    oof = {"linear": np.zeros(len(F)), "lambdamart": np.zeros(len(F)), "lambdamart_unconstrained": np.zeros(len(F))}
    for k in sorted(F.fold.unique()):
        tr, te = F.fold.ne(k), F.fold.eq(k)
        oof["linear"][te.to_numpy()] = LinearRanker().fit(F[tr]).score(F[te])
        oof["lambdamart"][te.to_numpy()] = LambdaMART().fit(F[tr]).score(F[te])
        oof["lambdamart_unconstrained"][te.to_numpy()] = LambdaMART(constrained=False).fit(F[tr]).score(F[te])
        print(f"fold {k} done", flush=True)
    results = {}
    for name, s in oof.items():
        r = case_ranks(F, s, base)
        results[name] = {"identifier": summarise(r.rank_id), "concept": summarise(r.rank_concept)}
    return {"results": results, "oof": oof}


def run() -> dict:
    F = pd.read_parquet(config.MODELS / "features_dev.parquet")
    base = pd.read_parquet(config.MODELS / "base_ranks_dev.parquet").set_index("case_id")
    split = pd.read_parquet(config.EVAL / "case_split.parquet").set_index("case_id")
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet")
    concept_of = dict(zip(diseases.disease_id, diseases.mondo_id))
    F = attach_concepts(F, concept_of)
    cv = cross_validate(F, base, split.fold)
    lin_mrr = cv["results"]["linear"]["concept"]["mrr"]
    lm_mrr = cv["results"]["lambdamart"]["concept"]["mrr"]
    product = "lambdamart" if lm_mrr > lin_mrr + MARGIN else "linear"
    T = fit_temperature(F, cv["oof"][product])
    # Base rankers on the same cases, for the comparison table.
    base_res = {}
    for k in ["lr", "bma", "p2d", "d2p", "exact", "bm25", "lsa"]:
        base_res[k] = summarise(base[f"rank_{k}"])
    final_lin = LinearRanker().fit(F)
    final_lm = LambdaMART().fit(F)
    (config.MODELS / "fusion_linear.json").write_text(json.dumps(final_lin.to_json(), indent=2))
    final_lm.model.booster_.save_model(str(config.MODELS / "fusion_lambdamart.txt"))
    meta = {"product": product, "temperature": T,
            "selection_rule": f"linear unless LambdaMART concept MRR exceeds it by more than {MARGIN}; both constrained to keep evidence monotone",
            "monotone": MONOTONE,
            "cv": cv["results"], "base_rankers_dev": base_res, "n_dev_cases": int(base.shape[0]),
            "features": FEATURES}
    (config.MODELS / "fusion_meta.json").write_text(json.dumps(meta, indent=2))
    oof = F[["case_id", "disease_id", "concept", "label"]].assign(score=cv["oof"][product])
    oof.to_parquet(config.MODELS / "oof_dev.parquet", index=False)
    meta["browser"] = fit_browser(F, base, split.fold)
    (config.MODELS / "fusion_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def fit_browser(F: pd.DataFrame, base: pd.DataFrame, folds: pd.Series) -> dict:
    """The ontology only ranker used by the in browser engine.

    Linear and LambdaMART are compared with the same grouped cross validation and
    the same pre-registered rule as the product. The chosen model is exported as
    plain JSON (weights, or every tree) for the JavaScript engine.
    """
    B = F[in_browser_shortlist(F)].copy()
    has_gold = B.groupby("case_id").label.max()
    b = base.copy()
    b["in_shortlist"] = b.index.map(lambda c: bool(has_gold.get(c, 0)))
    B = B.assign(fold=B.case_id.map(folds))
    oof = {"linear": np.zeros(len(B)), "lambdamart": np.zeros(len(B))}
    for k in sorted(B.fold.unique()):
        tr, te = B.fold.ne(k), B.fold.eq(k)
        oof["linear"][te.to_numpy()] = LinearRanker(features=BROWSER_FEATURES).fit(B[tr]).score(B[te])
        oof["lambdamart"][te.to_numpy()] = LambdaMART(features=BROWSER_FEATURES).fit(B[tr]).score(B[te])
    results = {}
    for name, sc in oof.items():
        r = case_ranks(B, sc, b.reindex(B.case_id.unique()))
        results[name] = {"identifier": summarise(r.rank_id), "concept": summarise(r.rank_concept)}
    choice = "lambdamart" if results["lambdamart"]["concept"]["mrr"] > results["linear"]["concept"]["mrr"] + MARGIN else "linear"
    T = fit_temperature(B, oof[choice])
    if choice == "linear":
        out = LinearRanker(features=BROWSER_FEATURES).fit(B).to_json()
    else:
        m = LambdaMART(features=BROWSER_FEATURES).fit(B)
        out = {"kind": "lambdamart", "features": BROWSER_FEATURES, "trees": export_trees(m.model.booster_)}
    out["temperature"] = T
    (config.MODELS / "fusion_browser.json").write_text(json.dumps(out))
    return {"cv": results, "product": choice, "temperature": T}


def export_trees(booster) -> list[dict]:
    """LightGBM trees as nested dicts: split feature index, threshold, default direction, leaves."""
    dump = booster.dump_model()

    def node(n: dict) -> dict:
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        return {"f": n["split_feature"], "t": n["threshold"], "d": bool(n["default_left"]),
                "l": node(n["left_child"]), "r": node(n["right_child"])}

    return [node(t["tree_structure"]) for t in dump["tree_info"]]


class TreeEnsemble:
    """NumPy evaluation of exported trees, the reference for the JavaScript port.

    Each tree is flattened into arrays (feature, threshold, default direction,
    children, leaf value) and all rows descend together one level at a time.
    """

    def __init__(self, d: dict):
        self.features = d["features"]
        self.trees = d["trees"]
        self._flat = [self._flatten(t) for t in self.trees]

    @staticmethod
    def _flatten(tree: dict):
        feat, thr, dflt, left, right, val = [], [], [], [], [], []

        def add(n: dict) -> int:
            i = len(feat)
            feat.append(-1)
            thr.append(0.0)
            dflt.append(False)
            left.append(-1)
            right.append(-1)
            val.append(0.0)
            if "v" in n:
                val[i] = n["v"]
            else:
                feat[i], thr[i], dflt[i] = n["f"], n["t"], n["d"]
                left[i] = add(n["l"])
                right[i] = add(n["r"])
            return i

        add(tree)
        return (np.array(feat), np.array(thr), np.array(dflt), np.array(left), np.array(right), np.array(val))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(np.float64)
        rows = np.arange(len(X))
        out = np.zeros(len(X))
        for feat, thr, dflt, left, right, val in self._flat:
            node = np.zeros(len(X), dtype=np.int64)
            active = feat[node] >= 0
            while active.any():
                n = node[active]
                v = X[rows[active], feat[n]]
                go_left = np.where(np.isnan(v), dflt[n], v <= thr[n])
                node[active] = np.where(go_left, left[n], right[n])
                active = feat[node] >= 0
            out += val[node]
        return out


def load_browser():
    d = json.loads((config.MODELS / "fusion_browser.json").read_text())
    model = LinearRanker.from_json(d) if d["kind"] == "linear_pairwise" else TreeEnsemble(d)
    return model, float(d["temperature"])


def run_browser() -> dict:
    F = pd.read_parquet(config.MODELS / "features_dev.parquet")
    base = pd.read_parquet(config.MODELS / "base_ranks_dev.parquet").set_index("case_id")
    split = pd.read_parquet(config.EVAL / "case_split.parquet").set_index("case_id")
    diseases = pd.read_parquet(config.PROCESSED / "diseases.parquet")
    F = attach_concepts(F, dict(zip(diseases.disease_id, diseases.mondo_id)))
    meta = json.loads((config.MODELS / "fusion_meta.json").read_text())
    meta["browser"] = fit_browser(F, base, split.fold)
    (config.MODELS / "fusion_meta.json").write_text(json.dumps(meta, indent=2))
    return meta["browser"]


class BoosterRanker:
    def __init__(self, booster, features: list[str]):
        self.booster, self.features = booster, features

    def score(self, df: pd.DataFrame) -> np.ndarray:
        # One thread: deterministic, and safe in worker processes on small machines.
        return self.booster.predict(df[self.features].to_numpy(np.float64), num_threads=1)


def load_product():
    meta = json.loads((config.MODELS / "fusion_meta.json").read_text())
    if meta["product"] == "linear":
        model = LinearRanker.from_json(json.loads((config.MODELS / "fusion_linear.json").read_text()))
    else:
        booster = lgb.Booster(model_file=str(config.MODELS / "fusion_lambdamart.txt"))
        model = BoosterRanker(booster, FEATURES)
    return model, meta


if __name__ == "__main__":
    meta = run()
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=2))
