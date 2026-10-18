"""Optional language model writer, verified exactly like the deterministic composer.

Off by default. When ODYSSEY_LLM=anthropic and ANTHROPIC_API_KEY are set, the
answer to a free text question can be rewritten by a language model from the
retrieved passages only. The model must end every sentence with its sources in
square brackets. Its output is parsed into sentences and passed through the same
verifier; any sentence that fails is dropped, and if fewer than half pass the
deterministic answer is used instead. The evaluation never uses this writer.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .verifier import Sentence

SYSTEM = (
    "You answer questions for a clinical geneticist using only the numbered sources provided. "
    "Write at most four short sentences. End every sentence with the identifiers of the sources it uses "
    "in square brackets, for example [DIS|OMIM:154700|overview]. Never give treatment, dosing or management "
    "advice. Never state that a patient has a disease. If the sources do not answer the question, say so."
)
CITE = re.compile(r"\[([^\[\]]+)\]\s*$")
API_URL = "https://api.anthropic.com/v1/messages"


def enabled() -> bool:
    return os.environ.get("ODYSSEY_LLM") == "anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def build_prompt(question: str, sources: dict[str, str]) -> str:
    blocks = [f"<source id=\"{k}\">\n{v}\n</source>" for k, v in sources.items()]
    return "Sources:\n" + "\n".join(blocks) + f"\n\nQuestion: {question}"


def parse(text: str) -> list[Sentence]:
    out = []
    for raw in re.split(r"(?<=\])\s+|\n+", text.strip()):
        raw = raw.strip()
        if not raw:
            continue
        m = CITE.search(raw)
        cites = [c.strip() for c in m.group(1).split(",")] if m else []
        body = CITE.sub("", raw).strip()
        if body:
            out.append(Sentence(text=body, cites=cites, role="answer"))
    return out


def call_anthropic(prompt: str, model: str | None = None, max_tokens: int = 400) -> str:
    body = json.dumps({
        "model": model or os.environ.get("ODYSSEY_LLM_MODEL", "claude-sonnet-5-5"),
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return "".join(block.get("text", "") for block in data.get("content", []))


def write(question: str, sources: dict[str, str], client=call_anthropic) -> list[Sentence]:
    """Return candidate sentences; the caller verifies them."""
    return parse(client(build_prompt(question, sources)))
