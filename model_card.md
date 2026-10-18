# Model card

Odyssey Anchor AI combines one fixed probabilistic model, two similarity measures, two text rankers, two learned rankers, a learned passage ranker and a rule based recogniser. This card describes each one, the data it saw and how it behaves. All numbers are from the strict protocol: for each patient, annotations resting only on that patient's paper are removed before scoring.

## Intended use

Decision support for clinical geneticists and genomic medicine teams who are ranking candidate rare diseases and genes for a patient with a phenotype description, and for teaching. Not a medical device. Not for unsupervised use by patients. Not validated on real clinic notes.

## Components

### Phenotype likelihood ratio (`rank/scorers.py`)

A naive Bayes model over phenotype terms. For an observed term *t* and a disease *d*, the probability that a patient with *d* shows *t* is the largest of:

* `F[d, t]`: the frequency of the same or a more specific annotation;
* `DIR[d, a] · p(t) / p(a)`, for each ancestor *a* the disease is annotated with directly (the patient term refines the annotation);
* `λ · F[d, a] · p(t) / p(a)`, for each ancestor *a* the disease has anything beneath (a sibling match through a shared ancestor).

This is mixed with the background, `(1 − η) P + η p(t)`, and divided by `p(t)`. A NOT annotation covering *t* caps the ratio at 0.05. An excluded term *e* contributes `max((1 − 0.95 F[d, e]) / (1 − p(e)), 0.05)`. The disease score is the sum of log ratios.

Settings were chosen by grid search on 800 development cases, by mean reciprocal rank: λ ∈ {0.02, 0.05, 0.1, 0.2, 0.4}, η ∈ {0.005, 0.02, 0.05}. The surface is flat; λ = 0.1 and η = 0.02 were chosen (see `reports/lr_tuning.json`).

### Semantic similarity (`rank/scorers.py`)

Resnik similarity with annotation based information content, averaged in both directions:

* `p2d`: for each patient term, the most informative ancestor it shares with any annotation of the disease.
* `d2p`: for each annotation, the same against the patient's closure, weighted by annotation frequency.

The best match average is their mean.

### Text rankers over disease documents (`retrieval/bm25.py`, `retrieval/lsa.py`)

Each disease has a token document made of its name (twice), synonyms, Mondo definition and the labels of its annotated features. BM25 (k1 1.2, b 0.75) and a 192 dimension latent semantic space over sublinear tf idf score the patient's term labels against it. In strict mode the affected disease documents are rebuilt without the removed features and scored with the fitted statistics.

### Fused ranker, the product (`rank/fusion.py`)

* **Shortlist**: the union of the top 120 by likelihood ratio, 80 by best match average, 40 by disease to patient similarity, 50 by BM25 and 30 by latent semantic score (about 200 diseases). Values are rounded to three decimals and ties broken by index, so the cut is reproducible.
* **Features (25)**: likelihood ratio and its observed and excluded parts; the shares of terms for and against; NOT conflicts; excluded features the disease shows in at least half of patients; the three similarity scores; exact overlap; BM25; LSA; profile size; an X linked recessive disease in a female; an adult onset disease in a child; gaps to the best value for four signals; and log ranks for five signals.
* **Learners**: a linear pairwise ranker (logistic regression on feature differences between the true disease and 50 alternatives, 25 hardest by likelihood ratio and 25 random) and LambdaMART (LightGBM, 400 trees, 31 leaves, learning rate 0.05, truncation at 20).
* **Monotone constraints**: every feature has a fixed direction: it may only raise a disease's score (likelihood ratios, similarities, BM25, gaps), only lower it (contradictions, NOT conflicts, recorded absences of typical features, worse ranks, an X linked recessive disease in a female, an adult onset disease in a child), or move freely (profile size). LambdaMART uses LightGBM's advanced monotone constraints; the linear model bounds each weight's sign. See *Why the rankers are constrained* below.
* **Selection**: five fold cross validation grouped by cohort group, on development cases only. The preregistered rule chose the linear model unless LambdaMART beat it by more than 0.02 in concept mean reciprocal rank. An unconstrained LambdaMART is trained in the same folds for comparison and is never used.
* **Calibration**: a softmax over the shortlist with a temperature fitted by minimising the negative log likelihood of the true disease on out of fold scores. Concept probability is the sum over a concept's records.

### Browser ranker (`rank/fusion.py`, `web/src/engine.js`)

The same procedure restricted to the 19 features that need no text index, and to the ontology shortlist (likelihood ratio, best match average, disease to patient similarity). The same preregistered rule picks between linear and LambdaMART.

### Passage ranker (`retrieval/passages.py`)

Logistic regression with balanced class weights over eleven features: BM25, LSA, a link to a named disease, gene or phenotype, intent fit, organ system fit, title overlap, passage subtype, and whether the passage's disease is in the current differential. It is trained on the 295 development questions plus an in session version of each disease question that says "the leading candidate" and passes a context of five diseases. Candidate recall on development questions is 100%.

### Phenotype recogniser (`extract/recognizer.py`)

Rule based, so it has no training data. It uses a lexicon of 43,062 normalised phrases from HPO labels and synonyms, longest match with order insensitive matching for phrases of two to five tokens, clause scoped negation (26 cues before the mention, 10 after), family history cues reset by a self reference, and uncertainty cues. Ambiguous phrases resolve to a primary label, then an exact synonym, then the more specific term.

## Why the rankers are constrained

The first development evaluation showed the next best question making the differential worse. Answering "absent" for a feature the true disease almost never has lowered that disease's probability in 91% of cases, which is the opposite of what the evidence says.

The cause is a curation artifact in published cases. A cohort paper about a disease records which of that disease's typical features each patient lacked, so excluded features are disproportionately the true disease's own hallmarks. In the development features, the true disease shows on average 1.45 of the patient's excluded features in at least half of its patients, against 1.01 for the other shortlisted diseases. An unconstrained ranker learns that "a typical feature of this disease was recorded absent" is evidence for the disease. That is right for journal cases and wrong for a clinician who records the absence of a feature they have just checked, and wrong for every answer to a next best question.

The fix keeps the evidence's clinical meaning by construction: the rankers can weigh each signal, but cannot reverse its direction. The decision was made on development data, before the blind set was scored. The unconstrained model's cross validated accuracy is reported next to the product's, so the cost of the constraint is visible.

## Development results

Five fold cross validation grouped by cohort group, 7,089 development patients, strict protocol.

| Ranker | Concept top 1 | Concept top 10 | Concept MRR | Identifier top 1 | Identifier top 10 |
| --- | --- | --- | --- | --- | --- |
| **LambdaMART, constrained (product)** | **27.2%** | **45.3%** | **0.338** | **24.0%** | **41.6%** |
| Linear pairwise, constrained | 11.2% | 29.2% | 0.170 | 9.7% | 26.2% |
| LambdaMART, unconstrained (not used) | 31.4% | 50.5% | 0.382 | 28.3% | 48.0% |
| Browser: linear, constrained (chosen) | 26.7% | 47.6% | 0.340 | 23.2% | 43.7% |
| Browser: LambdaMART, constrained | 24.8% | 46.6% | 0.323 | 22.0% | 43.4% |

Fitted temperatures are 1.27 for the product and 0.76 for the browser ranker.

## Blind results

2,045 patients, 188 diseases never seen in development, scored once.

| Component | Top 1 | Top 10 | MRR |
| --- | --- | --- | --- |
| Product, Mondo concept | 22.7% | 41.7% | 0.300 |
| Product, exact identifier | 20.5% | 38.9% | 0.271 |
| Browser ranker, Mondo concept | 23.0% | 42.0% | 0.297 |
| Causal gene, ranked by its best disease (1,976 patients with a gene) | 31.5% | 50.7% | 0.384 |

Other blind results:

* **Calibration:** the probability of the leading concept has an expected calibration error of 0.046 and a Brier score of 0.134.
* **Passage ranker:** 92.3% hit at 1 and 100% recall at 3 on 403 blind questions.
* **Recogniser:** 93.4% precision and 99.0% recall on present features, 98.6% and 88.1% on absent features, with no family history read as the patient's over 4,090 distractors.

Full tables are in [evaluation.md](evaluation.md).

## Failure modes

* **Strict removal empties profiles.** For 22.8% of blind patients, every annotation of the true disease rests on the patient's paper; with nothing left to match, the product finds only 13.7% of them in the top 10, through Orphanet twins and definitions.
* **Profile size carries the most split gain (43%).** It works mainly as a normaliser for summed likelihood ratios. But the strict protocol shrinks the true disease's profile, so part of what the ranker learned about size may not transfer to use on unstripped patients.
* **The fused ranker barely beats its best input.** On the blind set the product's identifier top 10 is 38.9%, against 37.6% for BM25 and 37.2% for Resnik patient to disease similarity. The ontology only browser ranker matches the product at concept level.
* **Mixed or atypical inheritance.** Diseases whose inheritance is neither plainly dominant, recessive nor X linked reach only 9.5% top 10.
* **Sparse input.** With only three features, top 10 falls from 41.7% to 24.6%. Replacing every feature with its parent term gives 28.6%.
* **The recogniser is rule based.** It handles the phrasing in the lexicon and simple reorderings. Measured on notes rendered from structured cases, it misses 12% of absent features, mostly through negation cues outside its list. It has not been measured on real clinic notes.
