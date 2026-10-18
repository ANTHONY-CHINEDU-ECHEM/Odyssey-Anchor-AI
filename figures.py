"""Figures for the README and documentation, drawn from the stored reports.

Nothing here scores a case: every figure reads reports/evaluation_blind.json,
reports/blind_predictions.parquet, reports/evaluation_dev.json or the model files.
Colours follow one fixed order: the product in blue, comparisons in grey,
a contrasting series in orange.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402

BLUE, GREY, ORANGE, AQUA = "#2a78d6", "#a3adac", "#eb6834", "#1baf7a"
INK, INK2, RULE, PAPER = "#15201f", "#4b5958", "#d9dedd", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RULE, "grid.linestyle": ":", "grid.linewidth": 0.8,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})
NAMES = {"lr": "Likelihood ratio", "bma": "Resnik best match average", "p2d": "Resnik, patient to disease",
         "d2p": "Resnik, disease to patient", "exact": "Exact term overlap", "bm25": "BM25, disease text",
         "lsa": "Latent semantic, disease text"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=160)
    plt.close(fig)


def hbar(ax, labels, values, colors, fmt="{:.1%}", xmax=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    xmax = xmax or max(values) * 1.18
    ax.set_xlim(0, xmax)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt.format(v), va="center", fontsize=9, color=INK2)


def rankers(B: dict) -> None:
    H = B["headline"]
    rows = [("Fused ranker (product)", H["identifier"])] + [(NAMES[k], v) for k, v in H["base_rankers_identifier"].items()]
    rows.append(("Browser ranker, concept", H.get("browser_concept", {"top1": 0, "top10": 0})))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, k, title in [(axes[0], "top1", "Top 1"), (axes[1], "top10", "Top 10")]:
        colors = [BLUE if "product" in r[0] else (AQUA if "Browser" in r[0] else GREY) for r in rows]
        hbar(ax, [r[0] for r in rows], [r[1][k] for r in rows], colors)
        ax.set_title(f"{title} accuracy, blind set, strict protocol")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if k == "top10":
            ax.set_yticklabels([])
    save(fig, "rankers.png")


def leakage(B: dict) -> None:
    L = B["leakage"]
    ks = ["top1", "top3", "top10", "top50"]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - 0.19, [L["strict"][k] for k in ks], 0.36, color=BLUE, label="Strict: source paper removed (reported)")
    ax.bar(x + 0.19, [L["released"][k] for k in ks], 0.36, color=ORANGE, label="Released annotations (leaky)")
    for i, k in enumerate(ks):
        ax.text(i - 0.19, L["strict"][k] + 0.01, f"{L['strict'][k]:.0%}", ha="center", fontsize=9)
        ax.text(i + 0.19, L["released"][k] + 0.01, f"{L['released'][k]:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, ["Top 1", "Top 3", "Top 10", "Top 50"])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("The same blind patients against two knowledge bases")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "leakage.png")


def topk_curve(P: pd.DataFrame) -> None:
    ks = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    series = [("Fused ranker, Mondo concept", P.rank_concept, BLUE, 2.4), ("Fused ranker, identifier", P.rank_id, BLUE, 1.2),
              ("Likelihood ratio", P.rank_lr, ORANGE, 1.6), ("Resnik best match average", P.rank_bma, GREY, 1.6),
              ("BM25, disease text", P.rank_bm25, AQUA, 1.6)]
    for name, r, c, lw in series:
        ax.plot(ks, [(r <= k).mean() for k in ks], color=c, lw=lw, ls="--" if "identifier" in name else "-", label=name)
    ax.set_xlabel("Rank of the published diagnosis at or above")
    ax.set_ylabel("Share of blind patients")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title("Cumulative accuracy by rank")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "topk_curve.png")


def calibration(B: dict) -> None:
    bins = B["headline"]["calibration_top1"]["bins"]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    n = np.array([b["n"] for b in bins])
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=BLUE, lw=1.6)
    ax.scatter([b["confidence"] for b in bins], [b["accuracy"] for b in bins], s=20 + 180 * np.sqrt(n / n.max()),
               color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    c = B["headline"]["calibration_top1"]
    ax.set_title(f"Calibration of the leading candidate\nECE {c['ece']:.3f}, Brier {c['brier']:.3f}")
    ax.set_xlabel("Stated probability")
    ax.set_ylabel("Observed share correct")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "calibration.png")


def subgroups(B: dict) -> None:
    S = B["subgroups"]
    titles = {"profile_band": "Annotations left after strict removal", "obs_band": "Observed features",
              "age_band": "Age at last encounter", "inheritance": "Inheritance", "sex_group": "Sex"}
    keys = [k for k in titles if k in S]
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.5, 1.0 + 0.42 * sum(len(S[k]) for k in keys) + 0.5 * len(keys)))
    for ax, k in zip(np.atleast_1d(axes), keys):
        rows = S[k]
        labels = [f"{g} ({v['n']})" for g, v in rows.items()]
        hbar(ax, labels, [v["top10"] for v in rows.values()], [BLUE] * len(rows), xmax=1.0)
        ax.set_title(titles[k], fontsize=10.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Top 10 accuracy by subgroup, Mondo concept", x=0.02, ha="left", fontweight="bold")
    save(fig, "subgroups.png")


def robustness(B: dict) -> None:
    R = B["robustness"]
    names = {"unchanged": "As published", "drop_half": "Half the features dropped", "keep_three": "Only three features",
             "no_excluded": "Excluded features removed", "generalise": "Every feature one level more general",
             "add_noise": "Two unrelated features added", "from_note": "Rendered as a note, then extracted"}
    keys = [k for k in names if k in R]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    hbar(ax, [names[k] for k in keys], [R[k]["top10"] for k in keys], [BLUE if k == "unchanged" else GREY for k in keys], xmax=1.0)
    ax.set_title("Top 10 accuracy under degraded input, blind set")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "robustness.png")


def next_question(B: dict) -> None:
    O = B["next_question"]["oracles"]
    titles = {"disease_model": "Answers from a simulated patient with the true disease",
              "record": "Answers only where the published record documents them"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, oracle in zip(axes, ("disease_model", "record")):
        S = O[oracle]
        for k, c, name in [("information_gain", BLUE, "Expected information gain (product)"),
                           ("most_frequent", ORANGE, "Most frequent feature of the leader"), ("random", GREY, "Random askable feature")]:
            v = S[k]["mrr_by_round"]
            ax.plot(range(len(v)), v, color=c, lw=2, marker="o", label=f"{name}, {S[k]['informative_answers_mean']:.1f} of 5 answered")
        ax.set_xticks(range(6), ["start", "1", "2", "3", "4", "5"])
        ax.set_xlabel("Questions asked")
        ax.set_title(titles[oracle], fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Mean reciprocal rank")
    fig.suptitle(f"Asking the next best question: {B['next_question']['cases']} blind cases with half the features hidden",
                 x=0.02, ha="left", fontweight="bold")
    save(fig, "next_question.png")


def retrieval(B: dict) -> None:
    M = B["retrieval"]["methods"]
    names = {"bm25": "BM25", "lsa": "Latent semantic", "hybrid": "Hybrid, equal weights", "learned": "Learned ranker (product)",
             "learned_without_kind_fit": "Learned, without intent fit", "learned_without_ent_disease": "Learned, without disease link",
             "learned_without_lsa": "Learned, without semantic score"}
    keys = [k for k in names if k in M]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    hbar(ax, [names[k] for k in keys], [M[k]["hit1"] for k in keys], [BLUE if k == "learned" else GREY for k in keys], xmax=1.0)
    ax.set_title(f"Knowledge retrieval, hit at 1, {B['retrieval']['n']} blind questions")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "retrieval.png")


def verifier(B: dict) -> None:
    V = B["verifier"]
    names = {"number_changed": "A number changed", "identifier_swapped": "A disease identifier swapped",
             "citation_removed": "Citations removed", "wrong_passage": "Citation to another passage",
             "treatment_added": "Treatment advice added", "certainty_added": "Definitive diagnosis added",
             "gene_swapped": "A gene symbol swapped", "dose_added": "A dose added"}
    keys = [k for k in names if V["catch_rate"].get(k) is not None]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = [V["catch_rate"][k] for k in keys]
    hbar(ax, [f"{names[k]} ({V['corrupted'][k]})" for k in keys], vals, [BLUE if v >= 0.95 else ORANGE for v in vals], xmax=1.08)
    ax.set_title("Verifier: share of corrupted sentences rejected")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "verifier.png")


def importance() -> None:
    meta = json.loads((config.MODELS / "fusion_meta.json").read_text())
    if meta["product"] != "lambdamart":
        return
    b = lgb.Booster(model_file=str(config.MODELS / "fusion_lambdamart.txt"))
    gain = b.feature_importance("gain")
    feats = meta["features"]
    order = np.argsort(gain)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    hbar(ax, [feats[i] for i in order], list(gain[order] / gain.sum()), [BLUE] * len(order))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("What the fused ranker relies on: share of split gain")
    save(fig, "importance.png")


def knowledge() -> None:
    stats = json.loads((config.INDEX / "passages_stats.json").read_text())
    man = json.loads((config.PROCESSED / "MANIFEST.json").read_text())["tables"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    hbar(axes[0], ["Disease overviews and organ systems", "Phenotype terms", "Genes", "Disease groups"],
         [stats["DIS"], stats["HPO"], stats["GENE"], stats["GRP"]], [BLUE] * 4, fmt="{:,.0f}")
    axes[0].set_title(f"{stats['total']:,} retrieval passages")
    rows = [("Disease annotations", man["annotations"]["rows"]), ("Mondo disease classes", man["mondo_classes"]["rows"]),
            ("Phenotype terms", man["hpo_terms"]["rows"]), ("Annotated diseases", man["diseases"]["rows"]),
            ("Gene to disease links", man["disease_genes"]["rows"]), ("Published patients", man["cases"]["rows"])]
    hbar(axes[1], [r[0] for r in rows], [r[1] for r in rows], [GREY] * len(rows), fmt="{:,.0f}")
    axes[1].set_xscale("log")
    axes[1].set_xlim(1000, 2_000_000)
    axes[1].set_title("Public sources, rows (log scale)")
    save(fig, "knowledge_base.png")


def run() -> list[str]:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    B = json.loads((config.REPORTS / "evaluation_blind.json").read_text())
    P = pd.read_parquet(config.REPORTS / "blind_predictions.parquet")
    rankers(B)
    leakage(B)
    topk_curve(P)
    calibration(B)
    subgroups(B)
    robustness(B)
    next_question(B)
    retrieval(B)
    verifier(B)
    importance()
    knowledge()
    return sorted(p.name for p in config.FIGURES.glob("*.png"))


if __name__ == "__main__":
    print(run())
