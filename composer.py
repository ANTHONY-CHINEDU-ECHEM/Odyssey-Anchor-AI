"""Deterministic, cited composition of the copilot answer.

Every sentence is assembled from fields of the analysis and names the sources
it rests on. Three kinds of source exist:

  PATIENT        the redacted record: recorded present and absent features
  MODEL          the computed outputs: probabilities, ranks, question gains
  passage ids    knowledge base passages (disease, organ system, phenotype, gene)

The sources dictionary built here is exactly what the verifier checks against,
so a sentence can only pass if its words and numbers are really there.
"""
from __future__ import annotations

import re

from ..knowledge.build import frequency_phrase
from ..knowledge.ids import disease_overview_id, gene_id, term_id
from ..retrieval.passages import kind_fit
from ..retrieval.text import tokens
from .verifier import DECISION_SUPPORT, NO_PASSAGE, Sentence

POLICY_LINE = DECISION_SUPPORT


def pct(p: float) -> str:
    v = p * 100
    return f"{v:.0f}%" if v >= 1 else "under 1%"


def patient_source(observed: list[dict], excluded: list[dict], note: str = "") -> str:
    parts = [f"{len(observed)} recorded features and {len(excluded)} recorded absences.",
             "Recorded present: " + "; ".join(f"{t['label']} ({t['id']})" for t in observed) + "."]
    if excluded:
        parts.append("Recorded absent: " + "; ".join(f"{t['label']} ({t['id']})" for t in excluded) + ".")
    if note:
        parts.append(note)
    return " ".join(parts)


def model_source(differential: list[dict], questions: list[dict], names: dict[str, str]) -> str:
    lines = []
    for d in differential:
        lines.append(f"Rank {d['rank']}: {d['name']} ({d['disease_id']}) estimated probability {pct(d['probability'])}"
                     f" among the shortlisted candidates; {len(d['evidence']['matches'])} recorded features compared,"
                     f" {sum(m['verdict'] == 'supports' for m in d['evidence']['matches'])} supporting.")
    for q in questions:
        lines.append(f"Question {q['label']} ({q['term']}): expected gain {q['expected_gain_bits']} bits;"
                     f" present favours {names.get(q['if_present_favours'], q['if_present_favours'])};"
                     f" absent favours {names.get(q['if_absent_favours'], q['if_absent_favours'])}.")
    return " ".join(lines)


def compose_differential(differential: list[dict], questions: list[dict], panel: list[dict],
                         n_obs: int, n_exc: int, names: dict[str, str], max_items: int = 5) -> list[Sentence]:
    out = [Sentence(
        text=f"Leading candidates for {n_obs} recorded features and {n_exc} recorded absences.",
        cites=["PATIENT", "MODEL"], role="differential")]
    for d in differential[:max_items]:
        did = d["disease_id"]
        out.append(Sentence(
            text=f"{d['rank']}. {d['name']} ({did}), estimated probability {pct(d['probability'])} among the shortlisted candidates.",
            cites=["MODEL", disease_overview_id(did)], role="differential"))
        sup = [m for m in d["evidence"]["matches"] if m["verdict"] == "supports" and m.get("passage")]
        sup.sort(key=lambda m: -m["log_lr"])
        if sup:
            items, cites = [], ["PATIENT"]
            for m in sup[:4]:
                if m["kind"] == "exact":
                    items.append(m["patient_label"])
                else:
                    items.append(f"{m['patient_label']} (annotated as {m['disease_label']})")
                cites.append(m["passage"])
            out.append(Sentence(text="Supporting features: " + "; ".join(items) + ".",
                                cites=list(dict.fromkeys(cites)), role="evidence"))
        con = [c for c in d["evidence"]["contradictions"] if "excluded_label" in c]
        if con:
            items = [f"{c['excluded_label']} ({frequency_phrase(c['frequency'])})" if c.get("frequency")
                     else c["excluded_label"] for c in con[:3]]
            out.append(Sentence(text="Recorded absent but typical: " + "; ".join(items) + ".",
                                cites=list(dict.fromkeys(["PATIENT"] + [c["passage"] for c in con[:3]])), role="evidence"))
        chk = d["evidence"]["to_check"][:3]
        if chk:
            items = [c["label"] for c in chk]
            out.append(Sentence(text="Worth checking: " + "; ".join(items) + ".",
                                cites=list(dict.fromkeys(c["passage"] for c in chk)), role="evidence"))
    if questions:
        q = questions[0]
        a = names.get(q["if_present_favours"], q["if_present_favours"])
        b = names.get(q["if_absent_favours"], q["if_absent_favours"])
        out.append(Sentence(
            text=f"The most informative next question is whether {q['label'].lower()} is present: present favours {a}, absent favours {b}.",
            cites=["MODEL", term_id(q["term"])], role="question"))
    if panel:
        items = [f"{g['gene']} ({g['disease_names'][0]})" for g in panel[:5]]
        out.append(Sentence(text="Genes behind the leading candidates, in order: " + "; ".join(items) + ".",
                            cites=[gene_id(g["gene"]) for g in panel[:5]], role="genes"))
    out.append(Sentence(text=POLICY_LINE, cites=[], role="policy"))
    return out


SENTENCE = re.compile(r"(?<=[a-z0-9)%])\.\s+(?=[A-Z0-9])")


def sentences_of(text: str) -> list[str]:
    """Split passage text into sentences without breaking inside parentheses."""
    out, depth, start = [], 0, 0
    for m in re.finditer(r"[()]|\.\s+(?=[A-Z0-9])", text):
        tok = m.group()
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(text[start:m.start() + 1].strip())
            start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [x for x in out if x]


def list_items(text: str) -> list[str]:
    """Top level comma separated items of a feature list."""
    items, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        items.append("".join(cur).strip())
    return [i.rstrip(".") for i in items if i]


def _pick(sentences: list[str], prefix: str) -> str | None:
    return next((x for x in sentences if x.startswith(prefix)), None)


def compose_answer(question: str, hits, info: dict, label_of=lambda t: t) -> list[Sentence]:
    """Answer from the top passage, choosing the segment that matches the question's intent.

    inheritance        the passage's mode of inheritance line
    genes_of_disease   the passage's associated genes line
    feature_in_disease the feature's own entry in the organ system list, with its frequency
    define_feature     the definition of the phenotype term
    diseases_of_gene   the gene passage, up to the first eight diseases
    otherwise          the two sentences sharing most words with the question
    """
    if hits is None or len(hits) == 0:
        return [Sentence(text=NO_PASSAGE, cites=[], role="policy")]
    intent = info.get("intent", "other")
    # Answer from the first retrieved passage whose kind fits the intent, if any.
    fitting = [i for i, pid in enumerate(hits.pid) if kind_fit(intent, pid)]
    top = hits.iloc[fitting[0] if fitting else 0]
    sents = sentences_of(top.text)
    name = top.title.split(":")[0]
    out: list[Sentence] = []
    if intent in ("inheritance", "genes_of_disease"):
        prefix = "Mode of inheritance" if intent == "inheritance" else "Associated genes"
        # The first overview passage that states it (an Orphanet record may not; its OMIM twin may).
        for r in hits.itertuples():
            if not r.pid.endswith("|overview"):
                continue
            line = _pick(sentences_of(r.text), prefix)
            if line:
                out.append(Sentence(text=f"{r.title.split(':')[0]}: {line[0].lower() + line[1:]}", cites=[r.pid], role="answer"))
                break
        if not out and top.pid.endswith("|overview"):
            what = "mode of inheritance" if intent == "inheritance" else "associated gene"
            out.append(Sentence(text=f"{name}: no {what} is recorded in the knowledge base.", cites=[top.pid], role="answer"))
    elif intent == "feature_in_disease" and top.pid.startswith("DIS|") and not top.pid.endswith("|overview"):
        labels = [label_of(t).lower() for t in info.get("terms", [])]
        body = top.text.split(": ", 1)[1] if ": " in top.text else top.text
        listed, _, absent = body.partition(" Documented as absent: ")
        for item in list_items(listed):
            if any(item.lower().startswith(l) for l in labels):
                out.append(Sentence(text=f"In {name}, {item[0].lower() + item[1:]}.", cites=[top.pid], role="answer"))
        for item in list_items(absent):
            if any(item.lower().startswith(l) for l in labels):
                out.append(Sentence(text=f"In {name}, {item.lower()} is documented as absent.", cites=[top.pid], role="answer"))
        if not out:
            out.append(Sentence(text=f"{label_of(info['terms'][0]) if info.get('terms') else 'That feature'} is not listed "
                                     f"for {name} in this organ system.", cites=[top.pid], role="answer"))
    elif intent == "define_feature" and top.pid.startswith("HPO|"):
        body = sents[1:3] if len(sents) > 1 else sents
        for x in body:
            out.append(Sentence(text=f"{name}: {x}", cites=[top.pid], role="answer"))
    elif intent == "diseases_of_gene" and top.pid.startswith("GENE|"):
        head, _, rest = top.text.partition(": ")
        items = [i.strip().rstrip(".") for i in re.split(r"(?<=\)),\s+", rest) if i.strip()]
        listed = "; ".join(items[:8])
        more = "; and others listed in the gene passage" if len(items) > 8 else ""
        out.append(Sentence(text=f"{head}: {listed}{more}.", cites=[top.pid], role="answer"))
    if not out:
        q = set(tokens(question))
        for r in hits.head(2).itertuples():
            ss = sentences_of(r.text)
            if not ss:
                continue
            best = max(ss, key=lambda x: len(q & set(tokens(x))))
            out.append(Sentence(text=f"{r.title.split(':')[0]}: {best}", cites=[r.pid], role="answer"))
    return out
