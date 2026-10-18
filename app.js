// Odyssey Anchor AI interface.
// Published cases show the full Python product (fused ranker, knowledge retrieval,
// verified sentences) precomputed under the strict protocol. Written notes run live
// in this page on the browser engine (engine.js, recognizer.js).

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (p, d = 0) => (p * 100 < 1 && p > 0 ? "<1%" : `${(p * 100).toFixed(d)}%`);
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const readJSON = (id) => { try { return JSON.parse(document.getElementById(id).textContent); } catch { return null; } };

const TEACH = readJSON("data-teaching") || [];
const REPORT = readJSON("data-report") || {};
const VIGNETTES = readJSON("data-vignettes") || [];
const FREQ = { "HP:0040280": "obligate (100%)", "HP:0040281": "very frequent (80 to 99%)", "HP:0040282": "frequent (30 to 79%)",
               "HP:0040283": "occasional (5 to 29%)", "HP:0040284": "very rare (1 to 4%)" };
function freqPhrase(raw) {
  if (!raw) return "frequency not stated";
  if (FREQ[raw]) return FREQ[raw];
  if (raw.includes("/")) { const [n, m] = raw.split("/"); return `${n} of ${m} reported patients`; }
  if (raw.endsWith("%")) return `${raw} of reported patients`;
  return raw;
}

const state = { source: "teaching", caseIdx: 0, vm: null, sel: 0, engine: null, rec: null, live: { observed: [], excluded: [], findings: [] } };

// tabs -------------------------------------------------------------------------------------------
const VIEWS = ["case", "results", "leakage", "safety", "knowledge", "method"];
function show(view) {
  for (const v of VIEWS) {
    $(`#view-${v}`).hidden = v !== view;
    $(`#tab-${v}`).setAttribute("aria-selected", String(v === view));
  }
  try { localStorage.setItem("oa-view", view); } catch {}
}
document.querySelectorAll("nav.tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

// view model from a Python product result ------------------------------------------------------------
function vmFromProduct(t) {
  const r = t.result;
  const name = Object.fromEntries(r.differential.map((d) => [d.disease_id, d.name]));
  return {
    engine: "product",
    flags: r.guard.messages || [],
    patient: r.patient,
    differential: r.differential.map((d) => ({ ...d, evidence: d.evidence })),
    questions: r.questions.map((q) => ({ label: q.label, term: q.term, gain: q.expected_gain_bits, yes: name[q.if_present_favours] || q.if_present_favours,
                                         yes_p: q.if_present_probability, no: name[q.if_absent_favours] || q.if_absent_favours, no_p: q.if_absent_probability })),
    genes: r.genes.map((g) => ({ gene: g.gene, weight: g.weight, disease: g.disease_names[0] })),
    sentences: r.sentences, sources: r.sources || {}, verification: r.verification, timings: r.timings_ms,
    gold: t.gold, question: t.question,
  };
}

// view model from the browser engine ---------------------------------------------------------------------
function vmFromBrowser(findings, observed, excluded, sex, age) {
  const e = state.engine;
  const t0 = performance.now();
  const pat = e.patient(observed, excluded, sex, age);
  const t1 = performance.now();
  const out = pat.observed.length ? e.differential(pat, 10) : { differential: [], sig: null, shortlist: 0 };
  const t2 = performance.now();
  const qs = out.differential.length ? e.nextQuestions(pat, out.differential) : [];
  const t3 = performance.now();
  const genes = e.genePanel(out.differential);
  const term = (i) => ({ id: e.ids[i], label: e.labels[i] });
  const uniq = (status) => [...new Map(findings.filter((f) => f.status === status).map((f) => [f.hpo_id, { id: f.hpo_id, label: f.label }])).values()];
  const diff = out.differential.map((d) => ({ ...d, evidence: null }));
  const vm = {
    engine: "browser", pat, sig: out.sig, flags: [],
    patient: { observed: pat.observed.map(term), excluded: pat.excluded.map(term), uncertain: uniq("uncertain"), family: uniq("family") },
    differential: diff, questions: qs, genes, sources: {}, sentences: [],
    timings: { normalise: +(t1 - t0).toFixed(1), score_and_rank: +(t2 - t1).toFixed(1), discriminate: +(t3 - t2).toFixed(1) },
  };
  composeBrowser(vm);
  return vm;
}

function ensureEvidence(vm, i) {
  const d = vm.differential[i];
  if (d && !d.evidence && vm.engine === "browser") {
    const ev = state.engine.explain(vm.pat, vm.sig, d.row);
    d.evidence = ev;
  }
  return d && d.evidence;
}

function composeBrowser(vm) {
  // Plain sentences, each naming the record or the annotations it rests on.
  const S = [];
  const P = vm.patient;
  vm.sources.PATIENT = `Recorded present: ${P.observed.map((t) => t.label).join("; ")}. Recorded absent: ${P.excluded.map((t) => t.label).join("; ")}.`;
  if (!vm.differential.length) {
    S.push({ text: "No phenotype was recorded as present, so there is no differential to rank.", cites: ["PATIENT"], role: "differential" });
  } else {
    S.push({ text: `Leading candidates for ${P.observed.length} recorded features and ${P.excluded.length} recorded absences.`, cites: ["PATIENT"], role: "differential" });
    vm.differential.slice(0, 3).forEach((d, i) => {
      const ev = ensureEvidence(vm, i);
      const key = `ANN|${d.disease_id}`;
      vm.sources[key] = `HPO annotations for ${d.name} (${d.disease_id}).`;
      S.push({ text: `${d.rank}. ${d.name} (${d.disease_id}), estimated probability ${pct(d.probability)} among the shortlisted candidates.`, cites: ["MODEL"], role: "differential" });
      const sup = ev.matches.filter((m) => m.verdict === "supports").sort((a, b) => b.log_lr - a.log_lr).slice(0, 4);
      if (sup.length) {
        vm.sources[key] += " " + sup.map((m) => `${m.disease_label} (${freqPhrase(m.frequency)})`).join("; ") + ".";
        S.push({ text: "Supporting features: " + sup.map((m) => (m.kind === "exact" ? m.patient_label : `${m.patient_label} (annotated as ${m.disease_label})`)).join("; ") + ".", cites: ["PATIENT", key], role: "evidence" });
      }
    });
    if (vm.questions.length) {
      const q = vm.questions[0];
      S.push({ text: `The most informative next question is whether ${q.label.toLowerCase()} is present: present favours ${q.yes}, absent favours ${q.no}.`, cites: ["MODEL"], role: "question" });
    }
  }
  vm.sources.MODEL = "Browser engine: likelihood ratio and semantic similarity over the HPO annotations, ranked by the ontology only ranker and calibrated by temperature.";
  S.push({ text: "This is decision support for a clinical geneticist: any diagnosis needs genetic testing and specialist review.", cites: [], role: "policy" });
  vm.sentences = S;
}

// rendering: record ---------------------------------------------------------------------------------------
function renderChips() {
  const vm = state.vm;
  const live = state.source === "live";
  const chip = (t, cls) => `<span class="chip ${cls}"><span class="lab">${esc(t.label)}</span><span class="mono">${esc(t.id)}</span>${live ? `<button aria-label="Remove ${esc(t.label)}" data-rm="${esc(t.id)}" data-kind="${cls}">×</button>` : "<span></span>"}</span>`;
  $("#chips-present").innerHTML = vm.patient.observed.map((t) => chip(t, "present")).join("") || `<span class="empty">None recorded.</span>`;
  $("#chips-absent").innerHTML = vm.patient.excluded.map((t) => chip(t, "absent")).join("") || `<span class="empty">None recorded.</span>`;
  const other = [];
  if (vm.patient.family?.length) other.push(`<p class="eyebrow" style="margin:12px 0 6px">Family history, not used</p><div class="chips">${vm.patient.family.map((t) => `<span class="chip"><span class="lab">${esc(t.label)}</span><span class="mono">${esc(t.id)}</span><span></span></span>`).join("")}</div>`);
  if (vm.patient.uncertain?.length) other.push(`<p class="eyebrow" style="margin:12px 0 6px">Uncertain, confirm before use</p><div class="chips">${vm.patient.uncertain.map((t) => `<span class="chip"><span class="lab">${esc(t.label)}</span><span class="mono">${esc(t.id)}</span><span></span></span>`).join("")}</div>`);
  $("#chips-other").innerHTML = other.join("");
  $("#terms-aside").textContent = `${vm.patient.observed.length} present, ${vm.patient.excluded.length} absent (most specific terms)`;
  document.querySelectorAll("[data-rm]").forEach((b) => b.addEventListener("click", () => {
    const id = b.dataset.rm;
    state.live.observed = state.live.observed.filter((x) => x !== id);
    state.live.excluded = state.live.excluded.filter((x) => x !== id);
    runLive(false);
  }));
}

function renderMarked(text, findings) {
  const box = $("#marked");
  if (!findings) { box.hidden = true; $("#legend").hidden = true; return; }
  const spans = [...findings].sort((a, b) => a.start - b.start);
  let html = "", pos = 0;
  for (const f of spans) {
    if (f.start < pos) continue;
    html += esc(text.slice(pos, f.start));
    const title = `${f.label} (${f.hpo_id})${f.cue ? `, cue: ${f.cue}` : ""}`;
    html += `<mark class="${f.status}" title="${esc(title)}">${esc(text.slice(f.start, f.end))}</mark>`;
    pos = f.end;
  }
  html += esc(text.slice(pos));
  box.innerHTML = html;
  box.hidden = false;
  $("#legend").hidden = false;
}

// rendering: differential, questions, genes ------------------------------------------------------------------
function renderDiff() {
  const vm = state.vm;
  const flags = [...(vm.flags || [])];
  $("#flags").innerHTML = flags.map((m) => `<div class="flag ${/emergency|suicide|crisis/i.test(m) ? "stop" : ""}">${esc(m)}</div>`).join("");
  const goldConcept = vm.gold?.concept;
  $("#diff").innerHTML = vm.differential.map((d, i) => {
    const hit = goldConcept && d.concept === goldConcept;
    return `<li><button aria-pressed="${i === state.sel}" data-i="${i}">
      <span class="rk">${d.rank}</span><span class="nm">${esc(d.name)} ${hit && state.revealed ? '<span class="hit-badge">PUBLISHED</span>' : ""}</span><span class="pc">${pct(d.probability)}</span>
      <span class="meter"><i style="width:${Math.max(1, d.probability * 100).toFixed(1)}%"></i></span>
      <span class="sub"><span class="mono">${esc(d.disease_id)}</span>${d.members.length > 1 ? `<span>+ ${d.members.length - 1} linked record</span>` : ""}${d.genes.length ? `<span>${esc(d.genes.slice(0, 3).join(", "))}</span>` : ""}</span>
    </button></li>`;
  }).join("") || `<li class="empty">Record at least one phenotype as present to rank diseases.</li>`;
  $("#diff-aside").textContent = vm.engine === "product" ? "fused ranker, Mondo concepts" : "browser engine, ontology ranker";
  document.querySelectorAll("#diff button").forEach((b) => b.addEventListener("click", () => { state.sel = +b.dataset.i; renderDiff(); renderCandidate(); }));
}

function renderQuestions() {
  const vm = state.vm;
  const live = state.source === "live";
  $("#questions").innerHTML = vm.questions.slice(0, 4).map((q) => `<div class="q">
      <div class="row" style="justify-content:space-between"><span class="qt">${esc(q.label)} <span class="mono muted">${esc(q.term)}</span></span><span class="bits">${fmt(q.gain, 2)} bits</span></div>
      <div class="split"><div>If present: ${esc(q.yes)} (${pct(q.yes_p)})</div><div>If absent: ${esc(q.no)} (${pct(q.no_p)})</div></div>
      ${live ? `<div class="row"><button class="btn ghost" data-ans="yes" data-term="${esc(q.term)}">Present</button><button class="btn ghost" data-ans="no" data-term="${esc(q.term)}">Absent</button></div>` : ""}
    </div>`).join("") || `<p class="empty">No question would change the leading candidates.</p>`;
  document.querySelectorAll("[data-ans]").forEach((b) => b.addEventListener("click", () => {
    const t = b.dataset.term;
    if (b.dataset.ans === "yes") state.live.observed.push(t); else state.live.excluded.push(t);
    runLive(false);
  }));
  const max = Math.max(...vm.genes.map((g) => g.weight), 1e-9);
  $("#genes").innerHTML = vm.genes.slice(0, 8).map((g) => `<div class="gene"><b>${esc(g.gene)}</b><span class="small muted" title="${esc(g.disease)}">${esc(g.disease)}</span><span class="num small" style="text-align:right">${pct(g.weight)}</span><span></span><span class="gb"><i style="width:${(100 * g.weight / max).toFixed(1)}%"></i></span><span></span></div>`).join("") || `<p class="empty">No genes yet.</p>`;
}

// rendering: candidate and evidence -------------------------------------------------------------------------
const KIND_TEXT = { exact: "same term", "more specific": "patient term is more specific", "more general": "patient term is more general", related: "shared ancestor", unexplained: "no informative link" };
function linkGlyph(kind) {
  const c = kind === "unexplained" ? "var(--rule)" : "var(--sea)";
  const dash = kind === "related" ? "3 3" : kind === "unexplained" ? "1 3" : "0";
  return `<svg width="64" height="10" viewBox="0 0 64 10" aria-hidden="true"><line x1="2" y1="5" x2="56" y2="5" stroke="${c}" stroke-width="1.6" stroke-dasharray="${dash}"/><path d="M55 1.5 61 5 55 8.5" fill="none" stroke="${c}" stroke-width="1.6"/></svg>`;
}

function renderCandidate() {
  const vm = state.vm;
  const d = vm.differential[state.sel];
  const box = $("#cand");
  if (!d) { box.innerHTML = `<h2>Candidate</h2><p class="empty">Select a candidate to see its evidence.</p>`; return; }
  const ev = ensureEvidence(vm, state.sel);
  const matches = ev.matches.slice().sort((a, b) => b.log_lr - a.log_lr);
  box.innerHTML = `
    <div class="cand-head"><p class="eyebrow">Candidate ${d.rank} of ${vm.differential.length}</p><h3>${esc(d.name)}</h3>
      <div class="ids"><span class="mono">${esc(d.members.join(" · "))}</span>${d.concept && d.concept !== d.disease_id ? `<span class="mono">${esc(d.concept)}</span>` : ""}</div></div>
    <dl class="kv"><dt>Probability</dt><dd class="num">${pct(d.probability, 1)} among shortlisted candidates</dd>
      <dt>Inheritance</dt><dd>${esc((d.inheritance || []).join(", ") || "not stated")}</dd>
      <dt>Genes</dt><dd class="mono">${esc((d.genes || []).join(", ") || "none linked")}</dd></dl>
    <p class="eyebrow" style="margin:16px 0 4px">Evidence path: patient term, link, disease annotation</p>
    <div class="evgrid">${matches.map((m) => `<div class="ev ${m.verdict}">
        <div><div class="pt">${esc(m.patient_label)}</div><span class="lrv">log LR ${fmt(m.log_lr, 2)}</span></div>
        <div class="link">${linkGlyph(m.kind)}${esc(KIND_TEXT[m.kind])}</div>
        <div class="dz">${m.disease_label ? `${esc(m.disease_label)}<small>${esc(freqPhrase(m.frequency))}${m.kind === "related" ? `, via ${esc(m.via_label)}` : ""}</small><div class="fbar"><i style="width:${Math.round((m.freq || 0) * 100)}%"></i></div>` : `<small>not annotated to this disease</small>`}</div>
      </div>`).join("")}</div>
    ${ev.contradictions.length ? `<p class="eyebrow" style="margin:16px 0 6px">Recorded absent, yet typical of this disease</p><ul class="list-plain">${ev.contradictions.map((c) => `<li>${esc(c.excluded_label || c.observed_label)} ${c.freq ? `<span class="muted small">(${pct(c.freq)} of patients)</span>` : `<span class="muted small">(documented not to occur)</span>`}</li>`).join("")}</ul>` : ""}
    ${ev.to_check.length ? `<p class="eyebrow" style="margin:16px 0 6px">Worth checking</p><ul class="list-plain">${ev.to_check.map((c) => `<li>${esc(c.label)} <span class="muted small">${esc(freqPhrase(c.frequency))}</span></li>`).join("")}</ul>` : ""}`;
}

// rendering: summary with citations --------------------------------------------------------------------------
function renderSay() {
  const vm = state.vm;
  const cites = (s) => s.cites.map((c) => `<button class="cite" data-src="${esc(c)}">${esc(c.length > 34 ? c.slice(0, 32) + "…" : c)}</button>`).join("");
  $("#say").innerHTML = vm.sentences.map((s) => `<p class="${s.role === "policy" ? "policy" : ""}">${esc(s.text)}${cites(s)}</p>`).join("");
  if (vm.engine === "product") {
    const v = vm.verification;
    $("#say-aside").innerHTML = `<span class="verified">${v.passed} of ${v.total} sentences verified</span>`;
  } else {
    $("#say-aside").textContent = "browser summary, full verifier runs in the service";
  }
  $("#srcbox").hidden = true;
  document.querySelectorAll(".cite").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.src;
    const box = $("#srcbox");
    box.innerHTML = `<div class="eyebrow">${esc(k)}</div>${esc(vm.sources[k] || "Source text not included in this view.")}`;
    box.hidden = false;
  }));
  $("#timings").innerHTML = Object.entries(vm.timings || {}).map(([k, v]) => `<span>${esc(k)} ${v}</span>`).join("");
}

function renderAll() { renderChips(); renderDiff(); renderQuestions(); renderCandidate(); renderSay(); }

// teaching cases ------------------------------------------------------------------------------------------------
function loadTeaching(i) {
  state.caseIdx = i;
  state.sel = 0;
  state.revealed = false;
  const t = TEACH[i];
  state.vm = vmFromProduct(t);
  const sexWord = t.sex === "FEMALE" ? "female" : t.sex === "MALE" ? "male" : "sex not recorded";
  $("#case-meta").innerHTML = `
    <div><span class="mono">${esc(t.case_id)}</span></div>
    <div>${t.age_years !== null && t.age_years !== undefined ? `${fmt(t.age_years, 1)} years, ` : ""}${sexWord}. Source paper <span class="mono">${esc(t.pmid)}</span>; annotations resting only on it are removed.</div>
    <div class="muted">Question asked: ${esc(t.question || "none")}</div>
    <details class="reveal" id="reveal"><summary>Reveal the published diagnosis</summary>
      <div class="dx">${esc(t.gold.label)}</div><div class="mono small">${esc(t.gold.disease_id)}${t.gold.gene ? ` · ${esc(t.gold.gene)}` : ""}</div>
      <div class="small muted">Rank of its Mondo concept in this differential: ${t.rank_concept <= 10 ? t.rank_concept : "outside the top 10"}.</div></details>`;
  $("#reveal").addEventListener("toggle", (e) => { state.revealed = e.target.open; renderDiff(); });
  $("#record-aside").textContent = "published case, blind set";
  renderAll();
}

function fillCasePicker() {
  $("#case-pick").innerHTML = TEACH.map((t, i) => `<option value="${i}">${esc(t.title)}</option>`).join("");
  $("#case-pick").addEventListener("change", (e) => loadTeaching(+e.target.value));
}

// live notes ----------------------------------------------------------------------------------------------------
function setSource(src) {
  state.source = src;
  $("#src-teaching").setAttribute("aria-pressed", String(src === "teaching"));
  $("#src-live").setAttribute("aria-pressed", String(src === "live"));
  $("#teaching-box").hidden = src !== "teaching";
  $("#live-box").hidden = src !== "live";
  $("#term-search").hidden = src !== "live";
  if (src === "teaching") loadTeaching(state.caseIdx);
  else runLive(true);
}

function runLive(fromNote) {
  const e = state.engine;
  if (!e) return;
  const text = $("#note").value;
  if (fromNote) {
    const { text: red, counts } = window.OA.redact(text);
    const findings = state.rec.extract(red);
    const terms = state.rec.toTerms(findings);
    state.live = { observed: terms.observed, excluded: terms.excluded, findings, red, counts };
    renderMarked(red, findings);
  }
  const age = $("#age").value === "" ? null : +$("#age").value;
  state.sel = 0;
  state.vm = vmFromBrowser(state.live.findings || [], state.live.observed, state.live.excluded, $("#sex").value, age);
  const nRed = Object.values(state.live.counts || {}).reduce((a, b) => a + b, 0);
  if (nRed) state.vm.flags.push(`${nRed} identifier${nRed > 1 ? "s were" : " was"} redacted before analysis.`);
  if (/\b(dose|dosage|mg\/kg|treat|treatment|therapy|prescri)/i.test(text)) state.vm.flags.push("Odyssey Anchor AI supports differential diagnosis only. It does not advise on treatment, dosing or management.");
  if (/ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules)|system prompt/i.test(text)) state.vm.flags.push("The note contained an instruction aimed at the system. It was ignored.");
  $("#record-aside").textContent = "your note, analysed in this page";
  renderAll();
}

function fillVignettes() {
  $("#vignette").innerHTML = VIGNETTES.map((v, i) => `<option value="${i}">${esc(v.title)}</option>`).join("");
  const pick = (i) => { const v = VIGNETTES[i]; $("#note").value = v.note; $("#sex").value = v.sex; $("#age").value = v.age ?? ""; };
  $("#vignette").addEventListener("change", (e) => { pick(+e.target.value); runLive(true); });
  if (VIGNETTES.length) pick(0);
  $("#analyse").addEventListener("click", () => runLive(true));
}

function termSearch() {
  const input = $("#term-input"), list = $("#term-list");
  let hits = [];
  input.addEventListener("input", () => {
    const q = input.value.trim();
    if (q.length < 2 || !state.engine) { list.hidden = true; return; }
    hits = state.engine.searchTerms(q, 10);
    list.innerHTML = hits.map((h, i) => `<li role="option" data-i="${i}"><span>${esc(h.label)}</span><span class="mono muted">${esc(h.id)}</span></li>`).join("");
    list.hidden = !hits.length;
    list.querySelectorAll("li").forEach((li) => li.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      state.live.observed.push(hits[+li.dataset.i].id);
      input.value = ""; list.hidden = true;
      runLive(false);
    }));
  });
  input.addEventListener("blur", () => setTimeout(() => (list.hidden = true), 120));
}

// charts ----------------------------------------------------------------------------------------------------------
function hbars(rows, { max = 1, format = (v) => pct(v, 1), cls = () => "" } = {}) {
  return `<div class="hb">${rows.map((r) => `<span class="lb ${r.head ? "head" : ""}">${esc(r.label)}</span><span class="tr"><i class="${cls(r)}" style="width:${(100 * Math.max(0, r.value) / max).toFixed(1)}%" data-tip="${esc(r.tip || `${r.label}: ${format(r.value)}`)}"></i></span><span class="vv">${format(r.value)}</span>`).join("")}</div>`;
}

function pairedBars(rows, keys, { max = 1, format = (v) => pct(v, 1) } = {}) {
  // rows: [{label, values: [a, b]}], keys: [{name, cls}]
  return `<div class="keyrow">${keys.map((k) => `<span class="${k.cls}">${esc(k.name)}</span>`).join("")}</div>
    <div class="hb">${rows.map((r) => r.values.map((v, j) => `<span class="lb ${j ? "" : "head"}">${j ? "" : esc(r.label)}</span><span class="tr"><i class="thin ${keys[j].cls === "k1" ? "hl" : keys[j].cls === "k3" ? "c3" : ""}" style="width:${(100 * v / max).toFixed(1)}%" data-tip="${esc(`${r.label}, ${keys[j].name}: ${format(v)}`)}"></i></span><span class="vv">${format(v)}</span>`).join("")).join("")}</div>`;
}

function lineChart(series, labels, { max = 1, step = null, w = 460, h = 220, format = (v) => pct(v, 0) } = {}) {
  const L = 40, R = 12, T = 12, B = 28;
  const x = (i) => L + (i * (w - L - R)) / (labels.length - 1);
  const y = (v) => T + (1 - v / max) * (h - T - B);
  const ticks = step ? Array.from({ length: Math.round(max / step) + 1 }, (_, i) => i * step) : [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">`;
  for (const t of ticks) svg += `<line class="grid" x1="${L}" x2="${w - R}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${format(t)}</text>`;
  labels.forEach((l, i) => (svg += `<text x="${x(i)}" y="${h - 8}" text-anchor="middle">${esc(l)}</text>`));
  for (const s of series) {
    svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}"/>`;
    s.values.forEach((v, i) => (svg += `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${s.color}" stroke="var(--sheet)" stroke-width="2" data-tip="${esc(`${s.name}, ${labels[i]}: ${format(v)}`)}"/>`));
  }
  return svg + "</svg>";
}

function reliability(bins, w = 300, h = 260) {
  const L = 40, R = 10, T = 10, B = 32;
  const x = (v) => L + v * (w - L - R), y = (v) => T + (1 - v) * (h - T - B);
  let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:380px" role="img">`;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) svg += `<line class="grid" x1="${L}" x2="${w - R}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t * 100}%</text><text x="${x(t)}" y="${h - 12}" text-anchor="middle">${t * 100}%</text>`;
  svg += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--ink-3)" stroke-dasharray="4 4"/>`;
  const maxN = Math.max(...bins.map((b) => b.n));
  svg += `<polyline fill="none" stroke="var(--bar)" stroke-width="2" points="${bins.map((b) => `${x(b.confidence)},${y(b.accuracy)}`).join(" ")}"/>`;
  for (const b of bins) svg += `<circle cx="${x(b.confidence)}" cy="${y(b.accuracy)}" r="${3 + 6 * Math.sqrt(b.n / maxN)}" fill="var(--bar)" fill-opacity="0.75" stroke="var(--sheet)" stroke-width="2" data-tip="${esc(`stated ${pct(b.confidence)}, observed ${pct(b.accuracy)}, ${b.n} cases`)}"/>`;
  svg += `<text x="${(L + w - R) / 2}" y="${h - 1}" text-anchor="middle">stated probability of the leading candidate</text></svg>`;
  return svg;
}

// results workspace ---------------------------------------------------------------------------------------------
function renderResults() {
  const B = REPORT.blind;
  const el = $("#view-results");
  if (!B) { el.innerHTML = `<p class="lead">The blind set has not been scored yet.</p>`; return; }
  const H = B.headline, C = H.concept;
  const ci = (a) => `95% interval ${pct(a[0], 1)} to ${pct(a[1], 1)}`;
  const base = H.base_rankers_identifier;
  const NAMES = { lr: "Likelihood ratio", bma: "Resnik best match average", p2d: "Resnik, patient to disease", d2p: "Resnik, disease to patient", exact: "Exact term overlap", bm25: "BM25 over disease text", lsa: "Latent semantic, disease text" };
  const rk = [{ label: "Fused ranker (product)", value: H.identifier.top10, head: true, hl: true, t1: H.identifier.top1 }]
    .concat(Object.entries(base).map(([k, v]) => ({ label: NAMES[k], value: v.top10, t1: v.top1 })));
  const sg = B.subgroups;
  const SGN = { sex_group: "Sex", age_band: "Age at last encounter", obs_band: "Observed features", profile_band: "Annotations left after strict removal", inheritance: "Inheritance" };
  const rob = B.robustness;
  const ROBN = { unchanged: "As published", drop_half: "Half the features dropped", keep_three: "Only three features", no_excluded: "Excluded features removed", generalise: "Every feature one level more general", add_noise: "Two unrelated features added", from_note: "Rendered as a note, then extracted" };
  const nqMax = Math.ceil(Math.max(...Object.values(B.next_question.oracles).flatMap((o) => Object.values(o).flatMap((v) => v.mrr_by_round))) * 10) / 10;
  el.innerHTML = `
    <p class="lead">Scored <b>once</b> on ${H.concept.n.toLocaleString()} real published patients from ${REPORT.frozen?.blind_groups ?? "held out"} cohort groups whose diseases and genes were never seen in development, under the strict protocol: every annotation resting only on a patient's own paper is removed before that patient is scored. The set was frozen and hashed at ${esc(REPORT.frozen?.frozen_utc)} (SHA256 <span class="mono">${esc((REPORT.frozen?.blind_case_sha256 || "").slice(0, 12))}…</span>) before any index or model existed.</p>
    <div class="tiles">
      <div class="tile"><div class="eyebrow">Top 1, Mondo concept</div><div class="v">${pct(C.top1, 1)}</div><div class="ci">${ci(C.top1_ci95)}</div></div>
      <div class="tile"><div class="eyebrow">Top 10, Mondo concept</div><div class="v">${pct(C.top10, 1)}</div><div class="ci">${ci(C.top10_ci95)}</div></div>
      <div class="tile"><div class="eyebrow">Mean reciprocal rank</div><div class="v">${fmt(C.mrr, 3)}</div><div class="ci">interval ${fmt(C.mrr_ci95[0], 3)} to ${fmt(C.mrr_ci95[1], 3)}</div></div>
      <div class="tile"><div class="eyebrow">Causal gene in top 10</div><div class="v">${pct(H.gene.top10, 1)}</div><div class="ci">gene ranked by its best disease</div></div>
      <div class="tile"><div class="eyebrow">Browser engine, top 10</div><div class="v">${pct(H.browser_concept?.top10 ?? 0, 1)}</div><div class="ci">ontology only ranker in this page</div></div>
    </div>
    <h2 class="sect">Ranker comparison</h2>
    <div class="grid2">
      <div class="chart"><h3>Top 10 accuracy by ranker</h3><p class="cap">Exact disease identifier, blind set. Ties are split evenly.</p>${hbars(rk, { cls: (r) => (r.hl ? "hl" : "") })}</div>
      <div class="chart"><h3>Top 1 accuracy by ranker</h3><p class="cap">The same rankers at the first position.</p>${hbars(rk.map((r) => ({ ...r, value: r.t1 })), { max: Math.max(...rk.map((r) => r.t1)) * 1.1, cls: (r) => (r.hl ? "hl" : "") })}</div>
    </div>
    <h2 class="sect">Calibration</h2>
    <div class="grid2">
      <div class="chart"><h3>Is the stated probability honest?</h3><p class="cap">Probability of the leading concept against how often it was correct. Expected calibration error ${fmt(H.calibration_top1.ece, 3)}, Brier ${fmt(H.calibration_top1.brier, 3)}.</p>${reliability(H.calibration_top1.bins)}</div>
      <div class="chart"><h3>What limits accuracy</h3><p class="cap">Top 10 by how many annotations of the true disease remain after its source paper is removed.</p>${hbars(Object.entries(sg.profile_band).map(([k, v]) => ({ label: `${k} (${v.n})`, value: v.top10 })), { cls: () => "hl" })}
        <p class="cap" style="margin-top:10px">Cases whose disease keeps at least three annotations (${pct(H.diagnosable_subset.share)} of the set): top 1 ${pct(H.diagnosable_subset.concept.top1, 1)}, top 10 ${pct(H.diagnosable_subset.concept.top10, 1)}.</p></div>
    </div>
    <h2 class="sect">Subgroups</h2>
    <div class="grid2">${Object.entries(sg).filter(([k]) => k !== "profile_band").map(([k, rows]) => `<div class="chart"><h3>${esc(SGN[k])}</h3><p class="cap">Top 10 accuracy, Mondo concept. Groups under 20 cases are not shown.</p>${hbars(Object.entries(rows).map(([g, v]) => ({ label: `${g} (${v.n})`, value: v.top10 })), { cls: () => "hl" })}</div>`).join("")}</div>
    <h2 class="sect">Robustness</h2>
    <div class="chart"><h3>Top 10 under degraded input</h3><p class="cap">Every blind case scored again after each change. The last row runs the whole note pathway: the case rendered as prose, then read back by the recogniser.</p>${hbars(Object.entries(rob).map(([k, v]) => ({ label: ROBN[k] || k, value: v.top10, head: k === "unchanged" })), { cls: (r) => (r.head ? "hl" : "") })}</div>
    <h2 class="sect">Next best question</h2>
    <div class="grid2">${["disease_model", "record"].map((o) => { const S = B.next_question.oracles[o]; return `<div class="chart"><h3>${o === "disease_model" ? "Answers from a simulated patient with the true disease" : "Answers only where the published record documents them"}</h3>
      <p class="cap">Mean reciprocal rank after each question. Half of each patient's features hidden; ${B.next_question.cases} blind cases. ${o === "record" ? "Published cases document few findings, so most questions go unanswered." : "Every question is answered by sampling the disease's own annotation frequency."}</p>
      <div class="keyrow"><span class="k1">information gain, ${S.information_gain.informative_answers_mean.toFixed(1)} answered</span><span class="k3">most frequent, ${S.most_frequent.informative_answers_mean.toFixed(1)}</span><span>random, ${S.random.informative_answers_mean.toFixed(1)}</span></div>
      ${lineChart([{ name: "information gain", color: "var(--bar)", values: S.information_gain.mrr_by_round }, { name: "most frequent", color: "var(--bar-3)", values: S.most_frequent.mrr_by_round }, { name: "random", color: "var(--bar-2)", values: S.random.mrr_by_round }], ["start", "1", "2", "3", "4", "5"], { max: nqMax, step: 0.1, format: (v) => v.toFixed(1) })}</div>`; }).join("")}</div>
    <h2 class="sect">Extraction and retrieval</h2>
    <div class="grid2">
      <div class="chart"><h3>Reading notes</h3><p class="cap">Blind cases rendered as notes with synonyms, inverted phrasing, eight negation cues and family history distractors.</p>
        <div class="tbl-wrap"><table class="t"><tr><th></th><th class="n">Precision</th><th class="n">Recall</th></tr>
        <tr><td>Present features</td><td class="n">${pct(B.extraction.observed.precision, 1)}</td><td class="n">${pct(B.extraction.observed.recall, 1)}</td></tr>
        <tr><td>Absent features</td><td class="n">${pct(B.extraction.excluded.precision, 1)}</td><td class="n">${pct(B.extraction.excluded.recall, 1)}</td></tr>
        <tr><td>Present, ancestor tolerant</td><td class="n">${pct(B.extraction.observed_tolerant_precision, 1)}</td><td class="n">–</td></tr>
        <tr><td>Family history read as the patient's</td><td class="n" colspan="2">${pct(B.extraction.family_leak_rate, 1)} of ${B.extraction.family_distractors.toLocaleString()}</td></tr></table></div></div>
      <div class="chart"><h3>Knowledge retrieval, ${B.retrieval.n} blind questions</h3><p class="cap">Including in session versions that say "the leading candidate".</p>
        <div class="tbl-wrap"><table class="t"><tr><th>Method</th><th class="n">Hit@1</th><th class="n">Recall@3</th><th class="n">MRR</th></tr>
        ${Object.entries(B.retrieval.methods).map(([k, v]) => `<tr><td>${esc(k.replaceAll("_", " "))}</td><td class="n">${fmt(v.hit1)}</td><td class="n">${fmt(v.recall3)}</td><td class="n">${fmt(v.mrr)}</td></tr>`).join("")}</table></div></div>
    </div>`;
  attachTips(el);
}

function renderLeakage() {
  const B = REPORT.blind, el = $("#view-leakage");
  if (!B) { el.innerHTML = ""; return; }
  const L = B.leakage;
  const ks = ["top1", "top3", "top10", "top50"];
  el.innerHTML = `
    <div class="prose"><h2 class="sect">Why the strict protocol</h2>
    <p>Disease annotations in the Human Phenotype Ontology are curated from published case reports, and the same reports are where the phenopacket store takes its patients from. In this release ${pct(REPORT.leak_stats?.cases_pmid_cited ?? 0.75)} of patients come from a paper that is also cited by the annotations. Scoring a patient against annotations taken from that patient's own description measures recall of the literature, not diagnosis.</p>
    <p>Under the strict protocol, before each patient is scored, every annotation whose only reference is that patient's paper is removed and the affected diseases are rescored. Annotations backed by another paper or by OMIM stay. Published benchmarks rarely do this, so their numbers are closer to the right hand bars.</p></div>
    <div class="grid2">
      <div class="chart"><h3>Same blind patients, two knowledge bases</h3><p class="cap">Mondo concept accuracy of the same fused ranker.</p>${pairedBars(ks.map((k) => ({ label: k.replace("top", "Top "), values: [L.strict[k], L.released[k]] })), [{ name: "strict (reported)", cls: "k1" }, { name: "released annotations", cls: "k3" }])}</div>
      <div class="chart"><h3>Exact identifier</h3><p class="cap">Without merging OMIM and Orphanet records of one disease.</p>${pairedBars(ks.map((k) => ({ label: k.replace("top", "Top "), values: [L.strict_identifier[k], L.released_identifier[k]] })), [{ name: "strict (reported)", cls: "k1" }, { name: "released annotations", cls: "k3" }])}</div>
    </div>`;
  attachTips(el);
}

function renderSafety() {
  const B = REPORT.blind, el = $("#view-safety");
  if (!B) { el.innerHTML = ""; return; }
  const V = B.verifier, P = B.probes, Cq = B.copilot;
  const CN = { number_changed: "A number changed", identifier_swapped: "A disease identifier swapped", citation_removed: "Citations removed", wrong_passage: "A citation pointed at another passage", treatment_added: "Treatment advice added", certainty_added: "A definitive diagnosis added", gene_swapped: "A gene symbol swapped", dose_added: "A dose added" };
  el.innerHTML = `
    <p class="lead">Every sentence the copilot shows must cite the record, the model output or a knowledge passage, and must survive a verifier before it is displayed. Over ${Cq.runs} blind copilot runs, <b>${Cq.verified} of ${Cq.sentences}</b> composed sentences passed. Median run time ${fmt(Cq.median_total_ms, 0)} ms.</p>
    <div class="grid2">
      <div class="chart"><h3>Verifier against deliberate corruptions</h3><p class="cap">Verified sentences altered one way at a time; the share rejected. Unaltered control sentences pass ${pct(V.control_pass_rate, 1)}.</p>${hbars(Object.entries(V.catch_rate).filter(([, v]) => v !== null).map(([k, v]) => ({ label: `${CN[k] || k} (${V.corrupted[k]})`, value: v })), { cls: (r) => (r.value >= 0.95 ? "hl" : "c3") })}</div>
      <div class="chart"><h3>Safety probes, ${P.passed} of ${P.total} passed</h3><p class="cap">Treatment and dose requests, hidden instructions, crisis and emergency disclosures, demands for certainty, identifiers, negation and family history traps, malformed input, audit chain.</p>
        <div class="tbl-wrap"><table class="t"><tr><th>Probe</th><th>Area</th><th>Result</th></tr>${P.probes.map((p) => `<tr><td class="mono">${esc(p.id)}</td><td>${esc(p.area)}</td><td class="${p.passed ? "pass" : "fail"}">${p.passed ? "passed" : "failed"}</td></tr>`).join("")}</table></div></div>
    </div>`;
  attachTips(el);
}

function renderKnowledge() {
  const el = $("#view-knowledge");
  const K = REPORT.kb || {}, M = REPORT.manifest || {};
  const T = M.tables || {};
  const rows = [["Phenotype terms (HPO)", T.hpo_terms?.rows], ["Disease annotations", T.annotations?.rows], ["Annotated diseases", T.diseases?.rows],
    ["Mondo disease classes", T.mondo_classes?.rows], ["Gene to disease links", T.disease_genes?.rows], ["Published patients", T.cases?.rows]];
  el.innerHTML = `
    <p class="lead">All knowledge comes from public sources: the Human Phenotype Ontology and its disease annotations (release ${esc(M.hpoa_version || "")}), the Mondo disease ontology, and the GA4GH phenopacket store (release ${esc(M.phenopacket_release || "")}) for evaluation only. Treatment sentences are stripped at ingestion (${(M.treatment_sentences_removed_from_mondo || 0).toLocaleString()} removed).</p>
    <div class="grid2">
      <div class="chart"><h3>Sources</h3><div class="tbl-wrap"><table class="t">${rows.map(([a, b]) => `<tr><td>${a}</td><td class="n">${(b || 0).toLocaleString()}</td></tr>`).join("")}</table></div></div>
      <div class="chart"><h3>Retrieval passages, ${(K.total || 0).toLocaleString()}</h3><p class="cap">${(K.tokens || 0).toLocaleString()} tokens indexed by BM25 and latent semantic analysis.</p>${hbars([["Disease overviews and organ system profiles", K.DIS], ["Phenotype terms", K.HPO], ["Genes", K.GENE], ["Disease groups", K.GRP]].map(([l, v]) => ({ label: l, value: v || 0 })), { max: K.DIS || 1, format: (v) => v.toLocaleString(), cls: () => "hl" })}</div>
    </div>
    <h2 class="sect">Look up a disease</h2>
    <div class="kb-search"><input class="field" id="kb-q" placeholder="Disease name or identifier, for example Noonan or OMIM:154700" style="width:100%" aria-label="Search diseases"></div>
    <div class="kb-out" id="kb-out"><p class="empty">The browser engine is still loading.</p></div>`;
  const run = () => {
    const e = state.engine;
    const out = $("#kb-out");
    if (!e) return;
    const q = $("#kb-q").value.trim().toLowerCase();
    if (q.length < 3) { out.innerHTML = `<p class="empty">Type at least three characters.</p>`; return; }
    const hits = [];
    for (let d = 0; d < e.nd && hits.length < 6; d++) if (e.dNames[d].toLowerCase().includes(q) || e.dIds[d].toLowerCase() === q) hits.push(d);
    out.innerHTML = hits.map((d) => {
      const lo = e.dirPtr[d], hi = e.dirPtr[d + 1];
      const items = [];
      for (let k = lo; k < hi; k++) items.push([e.dirVal[k], e.labels[e.dirIdx[k]], e.freqVocab[e.freqCode[k]]]);
      items.sort((a, b) => b[0] - a[0]);
      return `<div class="chart"><h3 style="font-family:var(--display);font-size:17px">${esc(e.dNames[d])}</h3><p class="cap mono">${esc(e.dIds[d])} · ${esc(e.concept[d])} · ${esc(e.dGenes[d].join(", ") || "no gene")} · ${esc(e.dInh[d].join(", ") || "inheritance not stated")}</p>
        <p class="small">${items.slice(0, 18).map((x) => `${esc(x[1])} <span class="muted">(${esc(freqPhrase(x[2]))})</span>`).join("; ")}${items.length > 18 ? `; and ${items.length - 18} more` : ""}.</p></div>`;
    }).join("") || `<p class="empty">No disease matches.</p>`;
  };
  $("#kb-q").addEventListener("input", run);
  state.kbRun = run;
}

function renderMethod() {
  const el = $("#view-method");
  const steps = [
    ["Guard", "Redact NHS numbers, names, dates, postcodes, telephone numbers, emails and hospital numbers; route treatment, override, crisis, emergency, certainty and off topic requests to fixed paths."],
    ["Extract", "Longest match recognition over 43,000 HPO labels and synonyms, with negation scoped to the clause, family history, uncertainty and character spans."],
    ["Normalise", "Map obsolete and alternative identifiers to current HPO terms and keep only the most specific."],
    ["Score", "A Bayesian phenotype likelihood ratio over three match routes with excluded features and NOT annotations; Resnik similarity in both directions; exact overlap; BM25 and latent semantic similarity over disease documents. All 12,880 diseases, every time."],
    ["Fuse", "A shortlist from every base ranker, then LambdaMART over 25 features, trained only on development cases and constrained so that no piece of evidence can count in the wrong direction; probabilities from a softmax with a fitted temperature."],
    ["Group", "OMIM and Orphanet records of one Mondo concept merged; the concept's probability is the sum over its records."],
    ["Explain", "For each candidate, the ontology path from every patient term to the best disease annotation, contradictions, and characteristic features not yet recorded."],
    ["Discriminate", "The next best question by expected information gain over the leading candidates."],
    ["Genes", "A gene panel pooled across every record of each concept, weighted by probability."],
    ["Retrieve", "Graph and text retrieval over 112,371 passages with a learned ranker, using the current differential as context."],
    ["Compose and verify", "Every sentence cites its sources; the verifier checks existence, words, numbers, identifiers and genes, and rejects treatment and certainty."],
    ["Audit", "Each run is appended to a hash chained log of redacted inputs and outputs."],
  ];
  el.innerHTML = `<div class="prose"><h2 class="sect">How a case is analysed</h2><p>Odyssey Anchor AI is decision support for clinical geneticists working through a rare disease differential. It never names a diagnosis, never suggests treatment, and shows its working for every candidate.</p></div>
    <div class="pipeline">${steps.map(([a, b]) => `<div class="st"><div><b>${a}</b><span class="muted">${b}</span></div></div>`).join("")}</div>
    <div class="prose" style="margin-top:20px"><h2 class="sect">Limits</h2>
    <p>The published cases are real but they are research descriptions, not clinic notes, and the note pathway is tested on notes rendered from them. Annotations backed by OMIM summaries may still echo a patient's paper. Orphanet and OMIM records are merged by Mondo cross references. This page is a research and teaching tool, not a medical device.</p></div>`;
}

function attachTips(root) {
  const tip = $("#tip");
  root.querySelectorAll("[data-tip]").forEach((n) => {
    n.addEventListener("mousemove", (ev) => { tip.textContent = n.dataset.tip; tip.hidden = false; tip.style.left = `${ev.clientX + 12}px`; tip.style.top = `${ev.clientY + 12}px`; });
    n.addEventListener("mouseleave", () => (tip.hidden = true));
  });
}

// boot ---------------------------------------------------------------------------------------------------------
async function loadEngine() {
  const b64 = document.getElementById("data-bundle").textContent.trim();
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const bundle = JSON.parse(await new Response(stream).text());
  await new Promise((r) => setTimeout(r, 0));
  state.engine = new window.OA.Engine(bundle);
  state.rec = new window.OA.Recognizer(state.engine, bundle.lexicon);
  $("#src-live").disabled = false;
  $("#engine-status").textContent = `Browser engine ready: ${state.engine.n.toLocaleString()} phenotype terms, ${state.engine.nd.toLocaleString()} diseases.`;
  if (state.kbRun) state.kbRun();
}

function boot() {
  fillCasePicker();
  fillVignettes();
  termSearch();
  $("#src-teaching").addEventListener("click", () => setSource("teaching"));
  $("#src-live").addEventListener("click", () => setSource("live"));
  if (TEACH.length) loadTeaching(0);
  renderResults(); renderLeakage(); renderSafety(); renderKnowledge(); renderMethod();
  let view = "case";
  const hash = location.hash.slice(1);
  try { view = VIEWS.includes(hash) ? hash : localStorage.getItem("oa-view") || "case"; } catch {}
  show(VIEWS.includes(view) ? view : "case");
  loadEngine().catch((err) => { $("#engine-status").textContent = `The browser engine could not load: ${err.message}`; });
}
boot();
