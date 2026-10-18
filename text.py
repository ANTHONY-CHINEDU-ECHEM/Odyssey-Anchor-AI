"""Tokenisation, normalisation and a dictionary matcher shared by every text component."""
from __future__ import annotations

import re
from collections import defaultdict

TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOP = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the this to was were with "
    "which what who how does do did can could should would may might than then there their these those "
    "about part seen linked associated caused cause causes variants variant gene genes condition conditions "
    "meant mean means inherited inheritance".split()
)
# Words that carry meaning in clinical phrases and must not be dropped by the matcher.
KEEP_FOR_MATCHING = frozenset("no not without absent".split())


def stem(tok: str) -> str:
    """A deliberately light stemmer: plurals and a few clinical suffix variants."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    for suffix, repl in (("ies", "y"), ("ses", "s"), ("xes", "x"), ("ae", "a"), ("es", "e"), ("s", "")):
        if tok.endswith(suffix) and not tok.endswith("ss") and not tok.endswith("us") and not tok.endswith("is"):
            return tok[: -len(suffix)] + repl
    return tok


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower()):
        if drop_stop and t in STOP:
            continue
        out.append(stem(t))
    return out


def norm_key(text: str) -> tuple[str, ...]:
    """Order sensitive normal form of a phrase, used as a dictionary key."""
    return tuple(stem(t) for t in TOKEN.findall(text.lower()) if t not in ("of", "the", "a", "an"))


class Lexicon:
    """Longest match dictionary lookup over token windows.

    Entries map a phrase to one or more concept identifiers. Matching is exact on
    the normalised token sequence, and optionally order insensitive for two to
    five token phrases (so 'cleft palate' also matches 'palate, cleft').
    """

    def __init__(self, max_len: int = 10, bag_match: bool = True):
        self.exact: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.bag: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.max_len = max_len
        self.bag_match = bag_match

    def add(self, phrase: str, concept: str) -> None:
        key = norm_key(phrase)
        if not key or len(key) > self.max_len:
            return
        self.exact[key].add(concept)
        if self.bag_match and 2 <= len(key) <= 5:
            self.bag[tuple(sorted(key))].add(concept)

    def find(self, text: str) -> list[dict]:
        """Return non overlapping longest matches with character spans."""
        spans = [(m.start(), m.end(), m.group().lower()) for m in TOKEN.finditer(text)]
        toks = [stem(s[2]) for s in spans]
        raw = [s[2] for s in spans]
        out, i = [], 0
        while i < len(toks):
            hit = None
            for L in range(min(self.max_len, len(toks) - i), 0, -1):
                window = [t for t, r in zip(toks[i:i + L], raw[i:i + L]) if r not in ("of", "the", "a", "an")]
                if not window:
                    continue
                key = tuple(window)
                concepts = self.exact.get(key)
                how = "exact"
                if not concepts and self.bag_match and 2 <= len(key) <= 5:
                    concepts = self.bag.get(tuple(sorted(key)))
                    how = "reordered"
                if concepts:
                    hit = (L, concepts, how)
                    break
            if hit:
                L, concepts, how = hit
                out.append({
                    "start": spans[i][0], "end": spans[i + L - 1][1],
                    "text": text[spans[i][0]:spans[i + L - 1][1]],
                    "concepts": sorted(concepts), "match": how, "token_start": i, "token_end": i + L,
                })
                i += L
            else:
                i += 1
        return out
