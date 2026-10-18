// Clinical phenotype recognition in the browser: a port of
// odyssey_anchor/extract/recognizer.py with the same lexicon, the same
// longest match rule and the same negation, family and uncertainty cues.

const TOKEN = /[a-z0-9]+/gi;
const ARTICLES = new Set(["of", "the", "a", "an"]);

export function stem(tok) {
  if (tok.length <= 3 || /^\d+$/.test(tok)) return tok;
  for (const [suffix, repl] of [["ies", "y"], ["ses", "s"], ["xes", "x"], ["ae", "a"], ["es", "e"], ["s", ""]]) {
    if (tok.endsWith(suffix) && !tok.endsWith("ss") && !tok.endsWith("us") && !tok.endsWith("is")) {
      return tok.slice(0, tok.length - suffix.length) + repl;
    }
  }
  return tok;
}

export function normKey(text) {
  return (text.toLowerCase().match(TOKEN) || []).filter((t) => !ARTICLES.has(t)).map(stem);
}

const PRE_NEG = /\b(no|not|without|denies|denied|denying|absence of|absent|negative for|ruled out|rules out|free of|never had|never developed|lack of|lacks|lacking|did not have|does not have|did not show|does not show|no evidence of|no history of|no signs of|nor|neither)\b/gi;
const POST_NEG = /^\W*(was|were|is|are|has been|have been)?\s*(absent|not seen|not observed|not present|excluded|ruled out|not detected|not found|not noted|negative)\b/i;
const FAMILY = /\b(mother|father|parent|parents|sibling|siblings|brother|sister|sisters|brothers|aunt|uncle|cousin|grandmother|grandfather|maternal|paternal|family history|relative|relatives|twin)\b/gi;
const UNCERTAIN = /\b(possible|possibly|suspected|suspect|query|questionable|may have|might have|probable|cannot exclude|could not exclude|unclear whether|equivocal)\b|\?/i;
const CLAUSE_BREAK = /\b(but|however|although|though|except|whereas|while)\b|[;:]/gi;
const SENTENCE = /(?<=[.!?])\s+|\n+/;
const SELF = /\b(the patient|patient|proband|he|she|they|the child|the boy|the girl|index case)\b/gi;

export class Recognizer {
  constructor(engine, lexicon, maxLen = 10) {
    this.e = engine;
    this.maxLen = maxLen;
    this.exact = new Map();
    this.bag = new Map();
    this.kind = new Map();
    for (const [phrase, idx, via] of lexicon) {
      const key = normKey(phrase);
      if (!key.length || key.length > maxLen) continue;
      const k = key.join(" ");
      if (!this.exact.has(k)) this.exact.set(k, new Set());
      this.exact.get(k).add(idx);
      if (key.length >= 2 && key.length <= 5) {
        const b = [...key].sort().join(" ");
        if (!this.bag.has(b)) this.bag.set(b, new Set());
        this.bag.get(b).add(idx);
      }
      this.kind.set(`${phrase}\u0001${idx}`, via);
    }
  }

  find(text) {
    const spans = [...text.matchAll(TOKEN)].map((m) => [m.index, m.index + m[0].length, m[0].toLowerCase()]);
    const toks = spans.map((s) => stem(s[2]));
    const raw = spans.map((s) => s[2]);
    const out = [];
    let i = 0;
    while (i < toks.length) {
      let hit = null;
      for (let L = Math.min(this.maxLen, toks.length - i); L >= 1; L--) {
        const win = [];
        for (let j = i; j < i + L; j++) if (!ARTICLES.has(raw[j])) win.push(toks[j]);
        if (!win.length) continue;
        let concepts = this.exact.get(win.join(" "));
        let how = "exact";
        if (!concepts && win.length >= 2 && win.length <= 5) {
          concepts = this.bag.get([...win].sort().join(" "));
          how = "reordered";
        }
        if (concepts) { hit = [L, concepts, how]; break; }
      }
      if (hit) {
        const [L, concepts, how] = hit;
        out.push({ start: spans[i][0], end: spans[i + L - 1][1], text: text.slice(spans[i][0], spans[i + L - 1][1]),
                   concepts: [...concepts].sort((a, b) => (this.e.ids[a] < this.e.ids[b] ? -1 : 1)), match: how });
        i += L;
      } else i += 1;
    }
    return out;
  }

  choose(text, concepts) {
    let best = concepts[0], bestKey = null;
    for (const c of concepts) {
      const via = this.kind.get(`${text.toLowerCase()}\u0001${c}`);
      const order = via === undefined ? 1 : via;
      const key = [order, -this.e.ic[c]];
      if (bestKey === null || key[0] < bestKey[0] || (key[0] === bestKey[0] && key[1] < bestKey[1])) { best = c; bestKey = key; }
    }
    const via = this.kind.get(`${text.toLowerCase()}\u0001${best}`);
    return [best, via === 0 ? "label" : "exact synonym"];
  }

  extract(text) {
    const findings = [];
    let pos = 0;
    for (const sentence of text.split(SENTENCE)) {
      if (!sentence.trim()) { pos += sentence.length + 1; continue; }
      let offset = text.indexOf(sentence, pos);
      if (offset < 0) offset = pos;
      pos = offset + sentence.length;
      const bounds = [0, ...[...sentence.matchAll(CLAUSE_BREAK)].map((m) => m.index), sentence.length];
      for (const m of this.find(sentence)) {
        const cstart = Math.max(...bounds.filter((b) => b <= m.start));
        const cend = Math.min(...bounds.filter((b) => b >= m.end));
        const before = sentence.slice(cstart, m.start);
        const after = sentence.slice(m.end, cend);
        let status = "present", cue = "";
        const head = sentence.slice(0, m.start);
        const fam = [...head.matchAll(FAMILY)];
        const selfref = [...head.matchAll(SELF)];
        if (fam.length && (!selfref.length || selfref[selfref.length - 1].index < fam[fam.length - 1].index)) {
          status = "family"; cue = fam[fam.length - 1][0];
        }
        const neg = [...before.matchAll(PRE_NEG)];
        const post = after.match(POST_NEG);
        if (status === "present" && (neg.length || post)) {
          status = "absent"; cue = neg.length ? neg[neg.length - 1][0] : post[0].trim();
        }
        if (status === "present") {
          const unc = before.match(UNCERTAIN) || after.slice(0, 12).match(UNCERTAIN);
          if (unc) { status = "uncertain"; cue = unc[0]; }
        }
        const [idx, via] = this.choose(m.text, m.concepts);
        findings.push({ hpo_id: this.e.ids[idx], label: this.e.labels[idx], text: m.text, start: offset + m.start,
                        end: offset + m.end, status, cue, match: m.match, via });
      }
    }
    return findings;
  }

  toTerms(findings) {
    const uniq = (a) => [...new Set(a)];
    const present = uniq(findings.filter((f) => f.status === "present").map((f) => f.hpo_id));
    const ps = new Set(present);
    return {
      observed: present,
      excluded: uniq(findings.filter((f) => f.status === "absent" && !ps.has(f.hpo_id)).map((f) => f.hpo_id)),
      uncertain: uniq(findings.filter((f) => f.status === "uncertain").map((f) => f.hpo_id)),
      family: uniq(findings.filter((f) => f.status === "family").map((f) => f.hpo_id)),
    };
  }
}

// Identifier redaction, mirroring odyssey_anchor/agent/guard.py.
const REDACTIONS = [
  ["EMAIL", /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/g],
  ["NHS_NUMBER", /\b\d{3}[ -]?\d{3}[ -]?\d{4}\b/g],
  ["PHONE", /(?:\+44\s?\d{2,4}|\b0\d{2,4})[\s-]?\d{3,4}[\s-]?\d{3,4}\b/g],
  ["HOSPITAL_NUMBER", /\b(?:MRN|hospital (?:number|no\.?)|hosp\.? no\.?|patient id)\s*[:#]?\s*[A-Z0-9]{5,}\b/gi],
  ["POSTCODE", /\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b/g],
  ["DATE", /\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b/gi],
  ["NAME", /\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Master|Prof)\.?\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)?/g],
  ["NAME", /\b(?:name|patient name|child's name)\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/gi],
];

export function redact(text) {
  const counts = {};
  for (const [label, re] of REDACTIONS) {
    text = text.replace(re, () => { counts[label] = (counts[label] || 0) + 1; return `[${label}]`; });
  }
  return { text, counts };
}
