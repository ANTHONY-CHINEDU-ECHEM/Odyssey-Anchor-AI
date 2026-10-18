# Architecture

![System architecture](../reports/figures/architecture.png)

Odyssey Anchor AI has two retrieval worlds that share one set of identifiers. The **ontology graph** ranks diseases from structured phenotypes, and the **passage store** answers questions in words. A disease, a phenotype term or a gene has the same identifier in both, so a sentence can cite a ranking, a disease annotation, a phenotype definition or a gene passage, and the verifier can check each of them in the same way.

## Layers

| Layer | Modules | What it holds or does |
| --- | --- | --- |
| Sources | `ingest/download.py` | Five public files downloaded from GitHub release assets, fingerprinted with SHA256 and release versions in `data/raw/MANIFEST.json` |
| Ingestion | `ingest/hpo.py`, `hpoa.py`, `mondo.py`, `genes.py`, `phenopackets.py`, `build.py` | Flat tables in `data/processed`, obsolete and alternative identifiers mapped, frequencies converted to smoothed probabilities, treatment sentences removed |
| Protocol | `eval/freeze.py` | Development and blind split, frozen and hashed before any index or model exists |
| Ontology | `ontology/graph.py`, `ontology/profiles.py` | Ancestor closure matrix, depth, information content, background frequency; direct, propagated and NOT disease profiles; the strict index |
| Knowledge | `knowledge/build.py`, `knowledge/ids.py` | 112,371 passages with stable identifiers, entities and references; one token document per disease |
| Retrieval | `retrieval/text.py`, `bm25.py`, `lsa.py`, `passages.py` | Tokeniser and lexicon matcher, BM25, latent semantic space, entity linker, intent detection, learned passage ranker |
| Extraction | `extract/recognizer.py`, `extract/render.py` | Phenotype recognition with negation, family history and uncertainty; note renderer for testing |
| Ranking | `rank/scorers.py`, `engine.py`, `features.py`, `fusion.py`, `tune.py`, `dataset.py` | Base rankers over every disease, shortlist, 25 features, LambdaMART product ranker, ontology only browser ranker, temperature calibration |
| Reasoning | `reason/evidence.py`, `discriminate.py`, `genes.py` | Evidence subgraphs, next best question, gene panel |
| Agent | `agent/guard.py`, `composer.py`, `verifier.py`, `writer_llm.py`, `audit.py`, `copilot.py` | The thirteen node state machine, from guard to audit |
| Evaluation | `eval/*.py` | Runner, metrics, perturbations, extraction, retrieval, verifier, probes, next question simulation, leakage, figures |
| Interfaces | `api/app.py`, `web/` | FastAPI service, single page interface, JavaScript browser engine with parity tests |

## Data flow

```
GitHub release assets ──► data/raw (+ MANIFEST.json)                         make data
        │
        ▼
parse, normalise, strip treatment ──► data/processed/*.parquet (+ MANIFEST.json)
        │
        ▼
freeze: cohort groups, 25% blind, SHA256 ──► data/eval/FROZEN.json           make freeze
        │
        ├─► passages ──► BM25 + LSA + learned ranker ──► data/index           make index
        ├─► disease documents ──► BM25 + LSA ──► data/index/disease_text.pkl
        │
        ▼
likelihood ratio settings on development cases ──► data/models/lr_params.json make tune
        │
        ▼
strict scoring of every development case ──► features_dev.parquet           make features
        │
        ▼
grouped cross validation, model choice, temperature ──► fusion_*.json/txt    make train
        │
        ▼
development evaluation ──► reports/evaluation_dev.json                       make evaluate
        │
        ▼
single blind run ──► reports/evaluation_blind.json, blind_predictions        make evaluate_blind
        │
        ▼
teaching cases, figures, browser bundle, interface                           make teaching figures bundle web
```

## The knowledge layer in detail

### Ontology graph

`Ontology` holds the 19,894 current HPO terms with integer indices. `ancestor_lists` is computed once in topological order, and `ANC` is a sparse boolean matrix with a one where column *j* is the row term or one of its ancestors. Two operations depend on it:

* **Closure of a patient**: the union of the ancestor lists of the observed terms.
* **Best shared ancestor**: for a patient term and an annotation, the common ancestor with the highest information content, with ties going to the deeper term.

### Disease profiles

`profiles.build` turns the annotation table into four matrices over terms, one row per disease:

| Matrix | Meaning |
| --- | --- |
| `DIR` | direct annotation frequency |
| `NOT` | explicit NOT annotations |
| `F` | propagated frequency: for term *t*, the highest frequency of any annotation at or below *t* |
| `C` | the support of `F` |

Information content is annotation based: `IC(t) = -log((n_t + 1) / (N + 1))`, where `n_t` is the number of diseases whose closure holds *t*. The background probability `p(t)` is the mean of `F[:, t]`, floored at `1/(4N)`.

### Strict index

`StrictIndex` records, for every PubMed identifier, the annotation rows whose only reference is that paper. `strict_profiles(pmid)` rebuilds the rows of the affected diseases without those annotations. The engine rescores just those diseases and substitutes them into every signal, including the text documents used by BM25 and LSA. Strict scoring therefore costs a few milliseconds, not a rebuild.

### Passage store

| Kind | Identifier | Count | Content |
| --- | --- | --- | --- |
| Disease overview | `DIS\|<disease>\|overview` | 12,880 | name, identifiers, synonyms, Mondo definition without treatment, inheritance, onset, genes, annotation count |
| Disease by organ system | `DIS\|<disease>\|<system term>` | 71,884 | every annotated feature in that organ system with its frequency phrase and references, plus features documented absent |
| Phenotype term | `HPO\|<term>` | 19,894 | label, definition, synonyms, parents, organ system, number of annotated diseases |
| Gene | `GENE\|<symbol>` | 5,277 | every linked disease with its inheritance |
| Disease group | `GRP\|<mondo>` | 2,436 | Mondo definition and member diseases for groups covering two to 400 annotated diseases |

Identifiers are fixed in `knowledge/ids.py` before the knowledge base exists, which is what allows the frozen questions to name their gold passages in advance.

## The runtime path

`Copilot.run` is a state machine of thirteen timed nodes. Each node's output is kept in the result, so the interface can show how an answer was produced.

1. **guard**: redaction runs before anything is logged, followed by rule based flags. Crisis and emergency block the analysis; treatment blocks the knowledge answer; override, certainty and off topic are recorded and shown.
2. **extract**: longest match over 43,062 phrases (labels, exact synonyms, plurals, layperson synonyms), order insensitive for two to five tokens. Negation is scoped to the clause, so "no seizures, hypotonia or ataxia" negates all three, and the scope ends at "but", "however" and similar words. Family history is reset by a self reference ("her mother is well and she has myopia").
3. **normalise**: identifiers are mapped to current terms and only the most specific terms are kept. An excluded term that is an ancestor of an observed one is dropped as contradictory.
4. **score**: every base ranker over all 12,880 diseases (see the model card).
5. **fuse**: the shortlist is the union of the top candidates of each base ranker. LambdaMART scores 25 features under monotone constraints, so no piece of evidence can count in the wrong direction (see the model card). A softmax with a fitted temperature gives probabilities.
6. **group**: records sharing a Mondo concept are merged, and the concept probability is the sum over its records.
7. **explain**: for each patient term, the explainer finds the best annotation and the ontology link between them. It also lists contradictions (absent features the disease shows in at least 30% of patients, and NOT annotations), plus characteristic features not yet recorded, deduplicated by ontology relation.
8. **discriminate**: the expected information gain of every askable phenotype over the posterior of the leading candidates.
9. **genes**: genes pooled across each concept's records, weighted by concept probability.
10. **retrieve**: candidates come from BM25 (top 100), LSA (top 50) and the graph route, meaning every passage attached to a named disease, gene or phenotype plus the current differential when the question refers to it. An eleven feature logistic ranker orders them.
11. **compose**: deterministic sentences, each with its sources. The answer to a question is taken from the first retrieved passage whose kind fits the intent (the inheritance line, the gene line, the feature's own frequency entry, or a definition). The optional language model writer drafts from the same passages.
12. **verify**: every sentence is checked against its cited sources; failures are dropped and reported.
13. **audit**: the redacted record of the run is appended to a hash chained log.

Typical timings on two CPU cores are about 80 ms for scoring all diseases, 120 ms for fusion, 90 ms for explanation and 130 ms for retrieval, with every other node under 10 ms.

## The browser engine

`web/src/engine.js` and `web/src/recognizer.js` port the recogniser, both scorers, the shortlist, the ontology only ranker, the explainer and the next best question to JavaScript. `odyssey_anchor/export/bundle.py` writes the bundle they need: terms, lexicon, diseases, direct and NOT annotations, frequency phrases and the ranker, in 1.8 MB of gzip. The JavaScript engine rebuilds the propagated matrices on load.

`web/tests/parity.test.mjs` checks the port against Python outputs:

* the same leading candidate for all 300 development cases;
* the same probabilities and likelihood ratios to within 1e-4;
* identical findings, spans and statuses on 150 rendered notes.

Values are rounded to three decimals before ranking in both languages. Diseases with identical annotation profiles therefore always tie, whatever floating point order produced their scores.

## Make targets

| Target | Writes |
| --- | --- |
| `make data` | `data/raw/*`, `data/processed/*.parquet`, both manifests |
| `make freeze` | `data/eval/case_split.parquet`, `questions.jsonl`, `FROZEN.json`, `blind_runs.json` |
| `make index` | `data/index/passages.parquet`, `passages_stats.json`, `disease_text.pkl`, `passage_index.pkl`, `data/models/passage_ranker.json` |
| `make tune` | `data/models/lr_params.json`, `reports/lr_tuning.json` |
| `make features` | `data/models/features_dev.parquet`, `base_ranks_dev.parquet` |
| `make train` | `data/models/fusion_*.json`, `fusion_lambdamart.txt`, `oof_dev.parquet` |
| `make evaluate` | `reports/evaluation_dev.json` |
| `make evaluate_blind` | `reports/evaluation_blind.json`, `blind_predictions.parquet`, a line in `data/eval/blind_runs.json` |
| `make teaching figures bundle web` | `reports/teaching_cases.json`, `reports/figures/*.png`, `web/data/*`, `web/dist*/` |
