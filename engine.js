// Odyssey Anchor AI browser engine.
//
// A port of the Python ontology scorers, the browser ranker, the evidence
// explainer and the next best question, run entirely client side from
// bundle.json.gz. Every number it produces is checked against Python outputs
// by web/tests/parity.test.mjs.

export class Engine {
  constructor(bundle) {
    this.b = bundle;
    const T = bundle.terms;
    this.ids = T.ids;
    this.labels = T.labels;
    this.parents = T.parents;
    this.ic = Float64Array.from(T.ic);
    this.bg = Float64Array.from(T.bg);
    this.pheno = Uint8Array.from(T.pheno);
    this.system = Int32Array.from(T.system);
    this.n = this.ids.length;
    this.index = new Map(this.ids.map((t, i) => [t, i]));
    this.params = bundle.params;
    this.model = bundle.model;
    const D = bundle.diseases;
    this.dIds = D.ids;
    this.dNames = D.names;
    this.concept = D.concept;
    this.dGenes = D.genes;
    this.dInh = D.inheritance;
    this.xr = Uint8Array.from(D.xr);
    this.adultOnly = Uint8Array.from(D.adult_only);
    this.nd = this.dIds.length;
    this.dIndex = new Map(this.dIds.map((d, i) => [d, i]));
    this._ancestors();
    this._profiles();
  }

  // ontology -----------------------------------------------------------------
  _ancestors() {
    const n = this.n;
    const indeg = new Int32Array(n);
    const children = Array.from({ length: n }, () => []);
    this.parents.forEach((ps, c) => ps.forEach((p) => { children[p].push(c); indeg[c] += 1; }));
    const queue = [];
    for (let i = 0; i < n; i++) if (indeg[i] === 0) queue.push(i);
    const order = [];
    while (queue.length) {
      const i = queue.pop();
      order.push(i);
      for (const c of children[i]) { indeg[c] -= 1; if (indeg[c] === 0) queue.push(c); }
    }
    this.anc = new Array(n);
    this.depth = new Int32Array(n);
    for (const i of order) {
      const acc = new Set([i]);
      let d = 0;
      for (const p of this.parents[i]) { for (const a of this.anc[p]) acc.add(a); d = Math.max(d, this.depth[p] + 1); }
      this.depth[i] = this.parents[i].length ? d : 0;
      this.anc[i] = Int32Array.from([...acc].sort((x, y) => x - y));
    }
  }

  isAncestor(a, t) {
    const arr = this.anc[t];
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < a) lo = m + 1; else hi = m; }
    return lo < arr.length && arr[lo] === a;
  }

  closure(idx) {
    const s = new Set();
    for (const i of idx) for (const a of this.anc[i]) s.add(a);
    return Int32Array.from([...s].sort((x, y) => x - y));
  }

  mostSpecific(idx) {
    const u = [...new Set(idx)].sort((x, y) => x - y);
    const s = new Set(u);
    const redundant = new Set();
    for (const i of u) for (const a of this.anc[i]) if (a !== i && s.has(a)) redundant.add(a);
    return u.filter((i) => !redundant.has(i));
  }

  // profiles: DIR and NOT from the bundle, F and C propagated here -------------------------
  _profiles() {
    const B = this.b.dir;
    this.dirPtr = Int32Array.from(B.indptr);
    this.dirIdx = Int32Array.from(B.indices);
    this.dirVal = Float64Array.from(B.data);
    this.freqCode = Int32Array.from(B.freq_code);
    this.freqVocab = B.freq_vocab;
    this.notPtr = Int32Array.from(this.b.not.indptr);
    this.notIdx = Int32Array.from(this.b.not.indices);
    const n = this.n;
    // Column lists (CSC) built from triplets.
    const colsDir = Array.from({ length: n }, () => []);
    const colsNot = Array.from({ length: n }, () => []);
    const colsF = Array.from({ length: n }, () => new Map());
    this.size = new Float64Array(this.nd);
    this.dirWeight = new Float64Array(this.nd);
    for (let d = 0; d < this.nd; d++) {
      const lo = this.dirPtr[d], hi = this.dirPtr[d + 1];
      this.size[d] = Math.log1p(hi - lo);
      for (let k = lo; k < hi; k++) {
        const s = this.dirIdx[k], f = this.dirVal[k];
        this.dirWeight[d] += f;
        colsDir[s].push([d, f]);
        for (const a of this.anc[s]) {
          const m = colsF[a];
          const prev = m.get(d);
          if (prev === undefined || f > prev) m.set(d, f);
        }
      }
      for (let k = this.notPtr[d]; k < this.notPtr[d + 1]; k++) colsNot[this.notIdx[k]].push(d);
    }
    const pack = (list) => ({ rows: Int32Array.from(list.map((x) => x[0])), vals: Float64Array.from(list.map((x) => x[1])) });
    this.colDir = colsDir.map(pack);
    this.colF = colsF.map((m) => {
      const rows = Int32Array.from([...m.keys()].sort((x, y) => x - y));
      return { rows, vals: Float64Array.from(rows, (r) => m.get(r)) };
    });
    this.colNot = colsNot.map((l) => Int32Array.from(l));
  }

  rowF(d) {
    // Propagated frequencies of one disease, as a Map term -> freq.
    const out = new Map();
    for (let k = this.dirPtr[d]; k < this.dirPtr[d + 1]; k++) {
      const s = this.dirIdx[k], f = this.dirVal[k];
      for (const a of this.anc[s]) { const p = out.get(a); if (p === undefined || f > p) out.set(a, f); }
    }
    return out;
  }

  // patient -------------------------------------------------------------------
  patient(observed, excluded = [], sex = "UNKNOWN_SEX", age = null) {
    const toIdx = (ts) => ts.map((t) => this.index.get(t)).filter((i) => i !== undefined);
    let obs = toIdx(observed).filter((i) => this.pheno[i]);
    obs = this.mostSpecific(obs);
    const closure = this.closure(obs);
    const cset = new Set(closure);
    const exc = [...new Set(toIdx(excluded).filter((i) => this.pheno[i] && !cset.has(i)))].sort((x, y) => x - y);
    return { observed: obs, excluded: exc, sex, age, closure };
  }

  // scorers ---------------------------------------------------------------------
  resnik(pat) {
    const nd = this.nd, ic = this.ic;
    const perTerm = pat.observed.map((t) => {
      const best = new Float32Array(nd);
      for (const a of this.anc[t]) {
        const col = this.colF[a];
        const v = ic[a];
        for (let k = 0; k < col.rows.length; k++) { const r = col.rows[k]; if (v > best[r]) best[r] = v; }
      }
      return best;
    });
    const p2d = new Float32Array(nd);
    if (perTerm.length) {
      for (let d = 0; d < nd; d++) { let s = 0; for (const row of perTerm) s += row[d]; p2d[d] = s / perTerm.length; }
    }
    // f[s]: best information content shared between term s and the patient closure.
    const inClosure = new Uint8Array(this.n);
    for (const c of pat.closure) inClosure[c] = 1;
    const f = new Float32Array(this.n);
    for (let s = 0; s < this.n; s++) {
      let m = 0;
      for (const a of this.anc[s]) if (inClosure[a] && ic[a] > m) m = ic[a];
      f[s] = m;
    }
    const d2p = new Float32Array(nd);
    for (let d = 0; d < nd; d++) {
      let s = 0;
      for (let k = this.dirPtr[d]; k < this.dirPtr[d + 1]; k++) s += this.dirVal[k] * f[this.dirIdx[k]];
      d2p[d] = s / Math.max(this.dirWeight[d], 1e-6);
    }
    const bma = new Float32Array(nd);
    for (let d = 0; d < nd; d++) bma[d] = 0.5 * (p2d[d] + d2p[d]);
    return { p2d, d2p, bma, perTerm };
  }

  termLikelihood(t) {
    const out = new Float64Array(this.nd);
    const { lam } = this.params;
    const F = this.colF[t];
    for (let k = 0; k < F.rows.length; k++) out[F.rows[k]] = F.vals[k];
    for (const a of this.anc[t]) {
      if (a === t) continue;
      const ratio = this.bg[t] / this.bg[a];
      const D = this.colDir[a];
      for (let k = 0; k < D.rows.length; k++) { const v = D.vals[k] * ratio; if (v > out[D.rows[k]]) out[D.rows[k]] = v; }
      const G = this.colF[a];
      for (let k = 0; k < G.rows.length; k++) { const v = lam * G.vals[k] * ratio; if (v > out[G.rows[k]]) out[G.rows[k]] = v; }
    }
    return out;
  }

  likelihoodRatio(pat) {
    const nd = this.nd, { eta, not_floor, exc_floor } = this.params;
    const obsLog = [];
    const conflicts = new Float32Array(nd);
    for (const t of pat.observed) {
      const pt = this.bg[t];
      const tl = this.termLikelihood(t);
      const hit = new Uint8Array(nd);
      for (const a of this.anc[t]) for (const r of this.colNot[a]) hit[r] = 1;
      const row = new Float32Array(nd);
      for (let d = 0; d < nd; d++) {
        let lr = ((1 - eta) * tl[d] + eta * pt) / pt;
        if (hit[d]) { lr = Math.min(lr, 1) * not_floor; conflicts[d] += 1; }
        row[d] = Math.log(lr);
      }
      obsLog.push(row);
    }
    const excLog = [];
    const excAgainst = new Float32Array(nd);
    for (const e of pat.excluded) {
      const col = new Float64Array(nd);
      const F = this.colF[e];
      for (let k = 0; k < F.rows.length; k++) col[F.rows[k]] = F.vals[k];
      const denom = 1 - Math.min(this.bg[e], 0.95);
      const row = new Float32Array(nd);
      for (let d = 0; d < nd; d++) {
        row[d] = Math.log(Math.max((1 - 0.95 * col[d]) / denom, exc_floor));
        if (col[d] >= 0.5) excAgainst[d] += 1;
      }
      excLog.push(row);
    }
    const lrObs = new Float32Array(nd), lrExc = new Float32Array(nd), lr = new Float32Array(nd);
    const support = new Float32Array(nd), against = new Float32Array(nd);
    const L2 = Math.log(2), LH = Math.log(0.5);
    for (let d = 0; d < nd; d++) {
      let so = 0, se = 0, sup = 0, ag = 0;
      for (const row of obsLog) { so += row[d]; if (row[d] > L2) sup += 1; if (row[d] < LH) ag += 1; }
      for (const row of excLog) se += row[d];
      lrObs[d] = so; lrExc[d] = se; lr[d] = so + se; support[d] = sup; against[d] = ag;
    }
    return { lr, lrObs, lrExc, support, against, conflicts, excAgainst, obsLog };
  }

  exactShare(pat) {
    const out = new Float32Array(this.nd);
    if (!pat.observed.length) return out;
    for (const t of pat.observed) for (const r of this.colF[t].rows) out[r] += 1;
    for (let d = 0; d < this.nd; d++) out[d] /= pat.observed.length;
    return out;
  }

  signals(pat) {
    const r = this.resnik(pat), l = this.likelihoodRatio(pat);
    const nObs = Math.max(pat.observed.length, 1);
    const female = pat.sex === "FEMALE";
    const child = pat.age !== null && pat.age !== undefined && pat.age < 16;
    const s = {
      lr: l.lr, lr_obs: l.lrObs, lr_exc: l.lrExc,
      support: l.support.map((x) => x / nObs), against: l.against.map((x) => x / nObs),
      not_conflicts: l.conflicts, exc_against: l.excAgainst,
      p2d: r.p2d, d2p: r.d2p, bma: r.bma, exact: this.exactShare(pat),
      size: Float32Array.from(this.size),
      xr_female: Float32Array.from(this.xr, (x) => (x && female ? 1 : 0)),
      adult_onset_child: Float32Array.from(this.adultOnly, (x) => (x && child ? 1 : 0)),
    };
    s.obsLog = l.obsLog;
    s.perTermIc = r.perTerm;
    return s;
  }

  // ranking -------------------------------------------------------------------------------
  static stableTop(arr, k) {
    // Rounded to three decimals with ties broken by index, as in odyssey_anchor/rank/features.py.
    const v = Float64Array.from(arr, (x) => Math.round(x * 1000) / 1000);
    const idx = Array.from(v.keys());
    idx.sort((a, b) => (v[b] - v[a]) || (a - b));
    return idx.slice(0, k);
  }

  static competitionRank(arr, i) {
    const v = arr[i];
    let c = 0;
    for (let k = 0; k < arr.length; k++) if (arr[k] > v) c++;
    return c + 1;
  }

  differential(pat, k = 10) {
    const sig = this.signals(pat);
    const pick = new Set();
    for (const [key, n] of Object.entries(this.params.shortlist)) for (const i of Engine.stableTop(sig[key], n)) pick.add(i);
    const rows = [...pick].sort((a, b) => a - b);
    const maxOf = (a) => { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; };
    const mx = { lr: maxOf(sig.lr), bma: maxOf(sig.bma) };
    // Ranks over all diseases for the log rank features.
    const rankArrays = {};
    // Ranks use values rounded to three decimals, as in odyssey_anchor/rank/features.py.
    const r3 = (v) => Math.round(v * 1000) / 1000;
    for (const key of ["lr", "bma", "d2p"]) {
      const sorted = Float64Array.from(sig[key], r3).sort();
      rankArrays[key] = (raw) => {
        const v = r3(raw);
        // number strictly greater than v, plus one
        let lo = 0, hi = sorted.length;
        while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= v) lo = m + 1; else hi = m; }
        return sorted.length - lo + 1;
      };
    }
    const M = this.model;
    const scores = rows.map((d) => {
      const fv = {};
      for (const f of M.features) {
        if (f.startsWith("gap_")) fv[f] = sig[f.slice(4)][d] - mx[f.slice(4)];
        else if (f.startsWith("logrank_")) fv[f] = Math.log(rankArrays[f.slice(8)](sig[f.slice(8)][d]));
        else fv[f] = sig[f][d];
      }
      return this._modelScore(M.features.map((f) => fv[f]));
    });
    const T = M.temperature;
    const top = Math.max(...scores);
    const ex = scores.map((s) => Math.exp((s - top) / T));
    const Z = ex.reduce((a, b) => a + b, 0);
    const groups = new Map();
    rows.forEach((d, i) => {
      const c = this.concept[d];
      const g = groups.get(c) || { concept: c, prob: 0, score: -Infinity, members: [] };
      g.prob += ex[i] / Z;
      g.members.push([scores[i], d]);
      if (scores[i] > g.score) g.score = scores[i];
      groups.set(c, g);
    });
    const list = [...groups.values()].sort((a, b) => (b.prob - a.prob) || (b.score - a.score));
    return {
      sig, shortlist: rows.length,
      differential: list.slice(0, k).map((g, i) => {
        const members = g.members.sort((a, b) => b[0] - a[0]).map((x) => x[1]);
        const rep = members[0];
        const genes = [...new Set(members.flatMap((m) => this.dGenes[m]))].sort();
        const inh = [...new Set(members.flatMap((m) => this.dInh[m]))].sort();
        return { rank: i + 1, concept: g.concept, probability: g.prob, score: g.score, row: rep,
                 disease_id: this.dIds[rep], name: this.dNames[rep], members: members.map((m) => this.dIds[m]),
                 genes, inheritance: inh };
      }),
    };
  }

  _modelScore(x) {
    const M = this.model;
    if (M.kind === "lambdamart") {
      // Exported LightGBM trees: go left when the value is at most the threshold (missing follows the default).
      let s = 0;
      for (const tree of M.trees) {
        let n = tree;
        while (n.v === undefined) {
          const v = x[n.f];
          n = (Number.isNaN(v) ? n.d : v <= n.t) ? n.l : n.r;
        }
        s += n.v;
      }
      return s;
    }
    let s = 0;
    for (let j = 0; j < x.length; j++) s += ((x[j] - M.mean[j]) / M.std[j]) * M.weights[j];
    return s;
  }

  // evidence -------------------------------------------------------------------------------
  explain(pat, sig, d) {
    const ic = this.ic, lo = this.dirPtr[d], hi = this.dirPtr[d + 1];
    const S = Array.from(this.dirIdx.slice(lo, hi));
    const Sf = Array.from(this.dirVal.slice(lo, hi));
    const Sc = Array.from(this.freqCode.slice(lo, hi));
    const matches = pat.observed.map((t, k) => {
      const mask = new Set(this.anc[t]);
      let best = [-1, -1, -1];
      for (const s of S) {
        let bestA = -1;
        for (const a of this.anc[s]) {
          if (!mask.has(a)) continue;
          if (bestA < 0 || ic[a] > ic[bestA] || (ic[a] === ic[bestA] && this.depth[a] > this.depth[bestA])) bestA = a;
        }
        if (bestA >= 0 && ic[bestA] > best[0]) best = [ic[bestA], s, bestA];
      }
      const [icA, s, a] = best;
      const lr = sig.obsLog[k][d];
      let kind;
      if (s < 0 || icA < 1) kind = "unexplained";
      else if (s === t) kind = "exact";
      else if (mask.has(s)) kind = "more specific";
      else if (this.isAncestor(t, s)) kind = "more general";
      else kind = "related";
      const item = { patient_term: this.ids[t], patient_label: this.labels[t], kind, log_lr: lr,
                     verdict: lr > Math.log(2) ? "supports" : lr < Math.log(0.5) ? "against" : "neutral" };
      if (s >= 0 && kind !== "unexplained") {
        const j = S.indexOf(s);
        Object.assign(item, { disease_term: this.ids[s], disease_label: this.labels[s], via_term: this.ids[a],
                              via_label: this.labels[a], freq: Sf[j], frequency: this.freqVocab[Sc[j]] });
      }
      return item;
    });
    const F = this.rowF(d);
    const contradictions = [];
    for (const e of pat.excluded) {
      const f = F.get(e) || 0;
      if (f >= 0.3) contradictions.push({ excluded_term: this.ids[e], excluded_label: this.labels[e], freq: f });
    }
    const known = new Set(pat.closure);
    for (const e of pat.excluded) for (const a of this.anc[e]) known.add(a);
    const related = new Set([...pat.observed, ...pat.excluded]);
    let check = [];
    S.forEach((s, j) => {
      const f = Sf[j];
      if (f < 0.4 || ic[s] < 3 || known.has(s)) return;
      for (const a of this.anc[s]) if (related.has(a)) return;
      check.push({ term: this.ids[s], label: this.labels[s], freq: f, frequency: this.freqVocab[Sc[j]], score: f * ic[s], i: s });
    });
    check.sort((x, y) => y.score - x.score);
    const chosen = [];
    check = check.filter((c) => {
      if (chosen.some((j) => this.isAncestor(c.i, j) || this.isAncestor(j, c.i))) return false;
      chosen.push(c.i);
      return true;
    }).slice(0, 6);
    return { matches, contradictions, to_check: check };
  }

  // next best question ----------------------------------------------------------------------
  nextQuestions(pat, diff, k = 5, minIc = 2.5, eta = 0.02) {
    const cands = diff.map((x) => x.row);
    let pi = diff.map((x) => x.probability);
    const Zp = pi.reduce((a, b) => a + b, 0);
    pi = pi.map((p) => p / Zp);
    const H = (p) => -p.reduce((acc, x) => acc + (x > 1e-12 ? x * Math.log2(x) : 0), 0);
    const known = new Set(pat.closure);
    const blocked = new Set(known);
    for (const e of pat.excluded) for (const a of this.anc[e]) blocked.add(a);
    const pool = new Set();
    for (const d of cands) for (let q = this.dirPtr[d]; q < this.dirPtr[d + 1]; q++) pool.add(this.dirIdx[q]);
    const rowsF = cands.map((d) => this.rowF(d));
    const out = [];
    for (const q of [...pool].sort((a, b) => a - b)) {
      if (blocked.has(q) || this.ic[q] < minIc || !this.pheno[q]) continue;
      let below = false;
      for (const a of this.anc[q]) if (a !== q && known.has(a) && this.ic[a] > 3) { below = true; break; }
      if (below) continue;
      const like = rowsF.map((m) => (1 - eta) * (m.get(q) || 0) + eta * this.bg[q]);
      const pYes = like.reduce((acc, l, j) => acc + l * pi[j], 0);
      const postY = like.map((l, j) => (l * pi[j]) / pYes);
      const postN = like.map((l, j) => ((1 - l) * pi[j]) / (1 - pYes));
      const eig = H(pi) - (pYes * H(postY) + (1 - pYes) * H(postN));
      const argmax = (a) => a.indexOf(Math.max(...a));
      out.push({ term: this.ids[q], label: this.labels[q], gain: eig, p_yes: pYes,
                 yes: diff[argmax(postY)].name, yes_p: Math.max(...postY),
                 no: diff[argmax(postN)].name, no_p: Math.max(...postN) });
    }
    return out.sort((a, b) => b.gain - a.gain).slice(0, k);
  }

  genePanel(diff, k = 8) {
    const score = new Map(), via = new Map();
    for (const d of diff) for (const g of d.genes) {
      score.set(g, (score.get(g) || 0) + d.probability);
      if (!via.has(g)) via.set(g, d.name);
    }
    return [...score.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1)).slice(0, k)
      .map(([g, w]) => ({ gene: g, weight: w, disease: via.get(g) }));
  }

  searchTerms(q, limit = 10) {
    const ql = q.toLowerCase();
    const hits = [];
    for (let i = 0; i < this.n; i++) {
      if (!this.pheno[i]) continue;
      const lab = this.labels[i].toLowerCase();
      if (lab.includes(ql) || this.ids[i].toLowerCase() === ql) hits.push([lab.startsWith(ql) ? 0 : 1, lab.length, i]);
    }
    hits.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return hits.slice(0, limit).map((h) => ({ id: this.ids[h[2]], label: this.labels[h[2]] }));
  }
}
