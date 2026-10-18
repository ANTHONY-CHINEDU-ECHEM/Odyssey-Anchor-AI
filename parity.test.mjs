// Parity between the browser engine and the Python reference.
//
// For 300 development cases the browser ranker must return the same leading
// concepts with the same probabilities, and the same likelihood ratio and
// semantic similarity for the true disease. For 150 rendered notes the
// recogniser must return the same findings, spans and statuses.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Engine } from "../src/engine.js";
import { Recognizer, redact, stem } from "../src/recognizer.js";

const here = dirname(fileURLToPath(import.meta.url));
const bundle = JSON.parse(gunzipSync(readFileSync(join(here, "../data/bundle.json.gz"))).toString());
const fixture = JSON.parse(readFileSync(join(here, "../data/parity_fixture.json")));
const engine = new Engine(bundle);
const rec = new Recognizer(engine, bundle.lexicon);

test("stemmer matches the Python rules", () => {
  assert.equal(stem("seizures"), "seizure");
  assert.equal(stem("anomalies"), "anomaly");
  assert.equal(stem("corpus"), "corpus");
  assert.equal(stem("ptosis"), "ptosis");
});

test("redaction removes identifiers", () => {
  const { text } = redact("Name: Amelia Jones, NHS 943 476 5919, DOB 12/03/2019, M40 5BG, a@b.com");
  for (const raw of ["Amelia", "943 476", "12/03/2019", "M40 5BG", "a@b.com"]) assert.ok(!text.includes(raw), raw);
});

test("differentials match Python on 300 development cases", () => {
  let sameTop1 = 0, sameTop10 = 0, maxProbDiff = 0, maxLrDiff = 0, maxBmaDiff = 0, compared = 0, close = 0;
  const n = fixture.cases.length;
  for (const c of fixture.cases) {
    const pat = engine.patient(c.observed, c.excluded, c.sex, c.age_years);
    assert.deepEqual(pat.observed.map((i) => engine.ids[i]), c.expected.observed_used);
    assert.deepEqual(pat.excluded.map((i) => engine.ids[i]), c.expected.excluded_used);
    const out = engine.differential(pat, 10);
    const got = out.differential.map((d) => d.concept);
    if (got[0] === c.expected.top_concepts[0]) sameTop1++;
    if (JSON.stringify(got) === JSON.stringify(c.expected.top_concepts)) sameTop10++;
    const g = engine.dIndex.get(c.disease_id);
    maxLrDiff = Math.max(maxLrDiff, Math.abs(out.sig.lr[g] - c.expected.gold_lr) / Math.max(1, Math.abs(c.expected.gold_lr)));
    maxBmaDiff = Math.max(maxBmaDiff, Math.abs(out.sig.bma[g] - c.expected.gold_bma));
    out.differential.forEach((d, i) => {
      if (d.concept === c.expected.top_concepts[i]) {
        const diff = Math.abs(d.probability - c.expected.top_probs[i]);
        maxProbDiff = Math.max(maxProbDiff, diff);
        compared += 1;
        if (diff < 1e-4) close += 1;
      }
    });
    assert.equal(out.shortlist, c.expected.shortlist_size);
  }
  console.log(`top1 agreement ${sameTop1}/${n}, full top 10 agreement ${sameTop10}/${n}, ` +
              `max probability difference ${maxProbDiff.toExponential(2)}, max relative lr difference ${maxLrDiff.toExponential(2)}, ` +
              `max bma difference ${maxBmaDiff.toExponential(2)}, probabilities within 1e-4: ${close}/${compared}`);
  assert.equal(sameTop1, n);
  // Diseases with identical annotation profiles tie exactly in principle; float32 sums in
  // Python and float64 sums here split such ties differently, which can swap two
  // near equal neighbours lower in the list. The leading candidate must always agree.
  assert.ok(sameTop10 >= Math.floor(0.95 * n), `full top 10 agreement ${sameTop10}/${n}`);
  // A tree split can fall either side of a threshold when float32 and float64 values straddle it,
  // so a few probabilities move slightly; nearly all must agree to 1e-4 and none may move by 1e-2.
  assert.ok(close >= 0.99 * compared, `probabilities within 1e-4: ${close}/${compared}`);
  assert.ok(maxProbDiff < 1e-2);
  assert.ok(maxLrDiff < 1e-4);
  assert.ok(maxBmaDiff < 1e-4);
});

test("extraction matches Python on 150 rendered notes", () => {
  let same = 0;
  for (const note of fixture.notes) {
    const got = rec.extract(note.text).map((f) => [f.hpo_id, f.start, f.end, f.status]);
    if (JSON.stringify(got) === JSON.stringify(note.findings)) same++;
  }
  console.log(`notes identical ${same}/${fixture.notes.length}`);
  assert.equal(same, fixture.notes.length);
});

test("evidence and next questions are well formed", () => {
  const c = fixture.cases[0];
  const pat = engine.patient(c.observed, c.excluded, c.sex, c.age_years);
  const out = engine.differential(pat, 8);
  const ev = engine.explain(pat, out.sig, out.differential[0].row);
  assert.equal(ev.matches.length, pat.observed.length);
  for (const m of ev.matches) assert.ok(["exact", "more specific", "more general", "related", "unexplained"].includes(m.kind));
  const qs = engine.nextQuestions(pat, out.differential);
  for (let i = 1; i < qs.length; i++) assert.ok(qs[i - 1].gain >= qs[i].gain);
  const panel = engine.genePanel(out.differential);
  assert.ok(panel.length > 0);
});
