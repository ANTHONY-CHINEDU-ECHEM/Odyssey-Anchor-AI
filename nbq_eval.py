"""Does asking the suggested question help? A simulation with a record oracle.

Half of each patient's observed features are hidden, along with every excluded
feature. The system then asks five questions, one at a time, under three
strategies:

  information gain   the next best question (reason/discriminate.py)
  most frequent      the most frequent unrecorded feature of the leading candidate
  random             a random askable feature of the leading candidates

Two oracles answer:

  record         the hidden published record: present if the question is the same
                 as or more general than a hidden observed feature (which is then
                 revealed), absent if a hidden excluded feature covers it, and
                 unknown otherwise, because a published case documents only some
                 findings
  disease model  a simulated patient with the true disease: present with the
                 disease's propagated annotation frequency for that feature
                 (smoothed to the background), from the released knowledge base,
                 drawn with a fixed seed; so every question gets an answer

The rank of the true disease is recorded after every question.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import zlib

import numpy as np
import pandas as pd

from .. import config
from ..rank.features import feature_frame
from ..reason.discriminate import next_questions
from . import runner

ROUNDS = 5
STRATEGIES = ("information_gain", "most_frequent", "random")
ORACLES = ("record", "disease_model")


def _differential(eng, pat, pmid):
    sig = eng.score(pat, pmid=pmid, strict=True)
    feats = feature_frame(eng, sig)
    feats["score"] = runner._W["model"].score(feats)
    z = np.exp((feats.score - feats.score.max()) / runner._W["T"])
    feats["prob"] = z / z.sum()
    feats["concept"] = feats.disease_id.map(runner._W["concept"])
    conc = feats.groupby("concept").agg(score=("score", "max"), prob=("prob", "sum")).sort_values(["prob", "score"], ascending=False)
    top = feats.sort_values("score", ascending=False).drop_duplicates("concept")
    top = top.assign(prob=top.concept.map(conc.prob)).sort_values("prob", ascending=False).head(10)
    return sig, conc, top


def _gold_rank(conc: pd.DataFrame, gold: str, fallback: int) -> int:
    if gold not in conc.index:
        return fallback
    return int((conc.prob > conc.loc[gold, "prob"]).sum()) + 1


def _ask(strategy: str, eng, sig, top: pd.DataFrame, asked: set, rng: random.Random) -> int | None:
    cands = top.disease_id.tolist()
    post = top.prob.to_numpy()
    if strategy == "information_gain":
        for q in next_questions(eng, sig, cands, post, k=12):
            i = eng.onto.index[q["term"]]
            if i not in asked:
                return i
        return None
    known = set(sig.patient.closure.tolist())
    pool = []
    for d in (cands[:1] if strategy == "most_frequent" else cands):
        prof, r = eng.profile_row(d, sig)
        lo, hi = prof.DIR.indptr[r], prof.DIR.indptr[r + 1]
        for t, f in zip(prof.DIR.indices[lo:hi], prof.DIR.data[lo:hi]):
            if t not in known and t not in asked and eng.ic[t] >= 2.5:
                pool.append((float(f * eng.ic[t]), int(t)))
    if not pool:
        return None
    if strategy == "most_frequent":
        return max(pool)[1]
    return rng.choice(pool)[1]


def simulate(case: dict) -> dict | None:
    eng = runner._W["eng"]
    obs = [t for t in case["observed"] if t in eng.onto.index]
    if len(obs) < 4:
        return None
    base_rng = random.Random(zlib.crc32(case["case_id"].encode()))
    base_rng.shuffle(obs)
    visible, hidden = obs[: len(obs) // 2], obs[len(obs) // 2:]
    gold = runner._W["concept"].get(case["disease_id"], case["disease_id"])
    gold_F = eng.prof.F[eng.row[case["disease_id"]]]
    gold_f = dict(zip(gold_F.indices.tolist(), gold_F.data.tolist()))
    out = {"case_id": case["case_id"]}
    for oracle in ORACLES:
        for strategy in STRATEGIES:
            rng = random.Random(zlib.crc32(f"{case['case_id']}|{oracle}|{strategy}".encode()))
            answer_rng = random.Random(zlib.crc32(f"{case['case_id']}|{oracle}".encode()))
            vis, exc, hid = list(visible), [], [eng.onto.index[t] for t in hidden]
            hid_exc = [eng.onto.index[t] for t in case["excluded"] if t in eng.onto.index]
            asked: set[int] = set()
            informative, ranks = 0, []
            for rnd in range(ROUNDS + 1):
                pat = eng.patient(vis, exc, case.get("sex", "UNKNOWN_SEX"), case.get("age_years"))
                sig, conc, top = _differential(eng, pat, case.get("pmid"))
                ranks.append(_gold_rank(conc, gold, len(conc) + 100))
                if rnd == ROUNDS:
                    break
                q = _ask(strategy, eng, sig, top, asked, rng)
                if q is None:
                    ranks += [ranks[-1]] * (ROUNDS - rnd)
                    break
                asked.add(q)
                if oracle == "record":
                    revealed = [h for h in hid if eng.onto.is_ancestor(q, h)]
                    if revealed:
                        vis += [eng.onto.ids[h] for h in revealed]
                        hid = [h for h in hid if h not in revealed]
                        informative += 1
                    elif any(eng.onto.is_ancestor(e, q) for e in hid_exc):
                        exc.append(eng.onto.ids[q])
                        informative += 1
                else:
                    p_yes = 0.98 * gold_f.get(q, 0.0) + 0.02 * float(eng.bg[q])
                    yes = answer_rng.random() < p_yes
                    (vis if yes else exc).append(eng.onto.ids[q])
                    informative += 1
                    # Behavioural check: how did this answer move the true disease's probability?
                    before = float(conc.prob.get(gold, 0.0))
                    pat2 = eng.patient(vis, exc, case.get("sex", "UNKNOWN_SEX"), case.get("age_years"))
                    after = float(_differential(eng, pat2, case.get("pmid"))[1].prob.get(gold, 0.0))
                    if strategy == "information_gain":
                        out.setdefault("answer_effects", []).append((bool(yes), after - before))
            out[f"{oracle}|{strategy}_ranks"] = ranks[: ROUNDS + 1]
            out[f"{oracle}|{strategy}_informative"] = informative
    return out


def run(cases: pd.DataFrame, n: int = 200, workers: int | None = None) -> dict:
    sample = cases[cases.n_observed >= 4].sample(n=min(n, len(cases)), random_state=config.SEED)
    recs = runner.records(sample)
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    with mp.get_context("spawn").Pool(workers, initializer=runner._init, initargs=(True,)) as pool:
        rows = [r for r in pool.imap(simulate, recs, chunksize=4) if r is not None]
    df = pd.DataFrame(rows)
    summary = {"cases": int(len(df)), "rounds": ROUNDS, "oracles": {}}
    for oracle in ORACLES:
        res = {}
        for s in STRATEGIES:
            R = np.array(df[f"{oracle}|{s}_ranks"].tolist(), dtype=float)
            res[s] = {
                "mrr_by_round": (1 / R).mean(axis=0).round(4).tolist(),
                "top10_by_round": (R <= 10).mean(axis=0).round(4).tolist(),
                "top1_by_round": (R <= 1).mean(axis=0).round(4).tolist(),
                "informative_answers_mean": float(df[f"{oracle}|{s}_informative"].mean()),
            }
        summary["oracles"][oracle] = res
    summary["strategies"] = summary["oracles"]["disease_model"]
    effects = [e for r in rows for e in r.get("answer_effects", [])]
    for label, want in (("present", True), ("absent", False)):
        d = np.array([x for yes, x in effects if yes == want])
        summary[f"answer_{label}"] = {"n": int(len(d)), "mean_change_in_true_disease_probability": float(d.mean()) if len(d) else None,
                                      "share_raising_it": float((d > 0).mean()) if len(d) else None,
                                      "share_lowering_it": float((d < 0).mean()) if len(d) else None}
    return summary
