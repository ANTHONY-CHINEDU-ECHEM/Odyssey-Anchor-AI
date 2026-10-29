# Odyssey Anchor AI

**Phenotype driven rare disease differential diagnosis with a verified graph retrieval copilot. A Bayesian and semantic ranker scores all 12,880 annotated diseases; a constrained learned ranker fuses and calibrates them; every candidate comes with its evidence path.**


<p align="center">
<img width="2400" height="1770" alt="architecture" src="https://github.com/user-attachments/assets/34944343-55e8-4080-a25c-e70d454c409c" />
</p>

SOURCE: <https://hpo.jax.org> · <https://mondo.monarchinitiative.org> · <https://github.com/monarch-initiative/phenopacket-store>

<p align="center">
<img width="447" height="335" alt="image" src="https://github.com/user-attachments/assets/eedb81cf-baa2-4e5a-8631-ef478e8d5a63" />
</p>


SOURCE: https://www.facebook.com/HealthyLivingMo/posts/the-zebra-is-the-official-symbol-of-rare-diseases-in-the-united-states-and-is-no/326766529492591/


**The next question that best separates the leaders, a gene panel, and answers drawn from a 112,371 passage knowledge base in which every sentence is cited and verified. Built entirely on public data and scored once, under a strict anti leakage protocol, on 2,045 real published patients whose diseases and genes were never seen during development.**

## Project brief

Most people with a rare disease wait years for a name. The phrase clinicians use is the diagnostic odyssey: a child with seizures, a heart defect and an unusual face is seen by neurology, cardiology and ophthalmology in turn, each recording a fragment, while the thousands of candidate conditions are individually so rare that no single clinician has met more than a handful. Exome and genome sequencing have changed what is possible, but sequencing returns hundreds of rare variants per person, and deciding which genes to look at first still depends on matching the patient's features to the right disease. That matching step, done by eye against textbooks and databases, is where days of a clinical geneticist's time go and where diagnoses are missed.

The knowledge needed for that match is public and unusually well structured. The Human Phenotype Ontology gives every clinical feature an identifier, a definition and a place in a hierarchy, so that "focal seizure" is understood to be a kind of "seizure". Its curated annotations link 12,880 OMIM, Orphanet and DECIPHER diseases to the features they cause, with how often and with the paper each link came from. The Mondo ontology says which OMIM and Orphanet records describe the same disease. And the GA4GH phenopacket store holds over ten thousand real patients from the published literature, each with the features they had, the features that were checked and found absent, and the confirmed diagnosis. Together they make it possible to build and honestly test a system that ranks diseases from a clinical description.

Odyssey Anchor AI is that system, built as a copilot a clinician can question. Given a free text note or a list of features, it reads the phenotypes (telling present from absent from family history), scores every annotated disease with a phenotype likelihood ratio and ontology similarity, fuses them with a learned ranker into calibrated probabilities, and for each candidate shows the ontology path from each patient feature to the disease annotation that explains it. It names the single next question that would best separate the leading candidates, proposes a gene panel, and answers knowledge questions from a 112,371 passage knowledge base through graph and text retrieval. Every sentence it shows cites its source and passes a verifier; it never names a diagnosis as fact and never advises on treatment.

The evaluation is built to be believed. Before any index or model existed, the published patients were split by cohort so that 188 diseases and their genes appear only in a blind set, and the blind set was hashed and scored exactly once. Most importantly, the disease annotations are themselves curated from the same papers the patients come from: 75% of patients come from a paper the annotations cite, and for one in five, every annotation of their disease rests on that paper alone. Scoring a patient against annotations copied from their own description measures recall of the literature, not diagnosis. Every number reported here therefore uses a strict protocol that removes, for each patient, every annotation resting only on that patient's paper, and the gap to the usual practice is reported alongside.

<p align="center">
<img width="1800" height="3811" alt="case" src="https://github.com/user-attachments/assets/7a8a6daf-9ff0-495f-ac86-5a96bf92525f" />
</p>

## What it does

| Workspace | What a clinician or reviewer can do |
| --- | --- |
| Case | Pick one of twelve published blind cases or write a note; see each phenotype highlighted as present, absent, family history or uncertain; read the differential grouped by disease, the evidence path from every recorded feature to the annotation that explains it, contradictions and features worth checking; answer the next best question and watch the differential update; read a gene panel and a summary in which every sentence cites its source |
| Blind results | Every result of the single blind run: rankers, calibration, subgroups, robustness, the next question simulation, extraction and retrieval |
| Leakage | The same patients scored against the released annotations and under the strict protocol, side by side |
| Safety | The verifier against eight kinds of corruption and the 25 safety probes |
| Knowledge | The public sources and the passage store, with a disease lookup over every annotation |
| Method | The thirteen step pipeline and its limits |

## Live demo

The whole system runs in the browser: the recogniser, both scorers, the ontology ranker, the explainer and the next best question were ported to JavaScript and checked against the Python outputs. The twelve published cases carry the full product run (fused ranker, knowledge retrieval, verified sentences) under the strict protocol.

To run the full service with the knowledge retrieval copilot, see Running it below.

## Screenshots

<p align="center">
<img width="694" height="1730" alt="evidence" src="https://github.com/user-attachments/assets/4eae092a-d6d8-4a73-a8b7-d9461e2b7f13" />
</p>



**Evidence path.** A blind case whose source paper has been removed from the knowledge base. The published diagnosis, Waardenburg syndrome type 3, ranks first at 57%. Each recorded feature is linked to the disease annotation that explains it, with the ontology link (same term, more specific, more general or a shared ancestor), the annotated frequency and the feature's log likelihood ratio. Features recorded absent that the disease usually shows are listed, as are the characteristic features not yet recorded.

<p align="center">
<img width="1800" height="2341" alt="note" src="https://github.com/user-attachments/assets/ae3dae6a-60a7-4ffb-8f67-4fcd384632f6" />
</p>

**A written note, analysed in the page.** Identifiers are redacted before anything else happens. Each phenotype is read as present, recorded absent, family history or uncertain, and the father's myopia is kept out of the patient's record. The differential, evidence and questions are computed in the browser.

<p align="center">
<img width="540" height="2199" alt="question_answered" src="https://github.com/user-attachments/assets/d8a166cd-3383-4560-8d7a-cbcfdece2c80" />
</p>


**Next best question.** Each suggestion shows its expected information gain in bits and which candidate a present or absent answer would favour. Answering updates the differential at once.

<p align="center">
<img width="694" height="1761" alt="summary" src="https://github.com/user-attachments/assets/2b18b949-2233-4bec-828b-b335d67ab72f" />
</p>


**Cited summary.** Every sentence names its sources: the patient record, the model output or a knowledge passage. Clicking a source shows its text. The verifier count sits in the header.

<p align="center">
<img width="1800" height="3734" alt="results" src="https://github.com/user-attachments/assets/b19568dd-e030-49d4-aace-486ce3caa735" />
</p>


**Blind results.** The single blind run, from the headline to calibration, subgroups, robustness and the next question simulation.

DARK THEME

<p align="center">
<img width="1800" height="1250" alt="dark" src="https://github.com/user-attachments/assets/d411e4dd-5eaf-4352-9064-e1d2a9497567" />
</p>


PHONE

<p align="center">
<img width="800" height="1720" alt="phone" src="https://github.com/user-attachments/assets/e7fac4c9-d66e-4e5e-a33f-8e075ebe9cf4" />
</p>

## Results

### Blind set, scored once

2,045 published patients in 170 cohort groups and 188 diseases that never appeared in development, frozen and hashed at 19:56 UTC on 17 October 2026 (SHA256 `ac018840…`) before any index or model existed, and scored once from 23:28 to 00:33 UTC. Every number uses the strict protocol, and intervals resample cohort groups.

| Fused ranker, blind | Top 1 | Top 3 | Top 10 | Top 50 | MRR |
| --- | --- | --- | --- | --- | --- |
| **Mondo concept** | **22.7%** (16.4% to 29.3%) | **34.2%** | **41.7%** (31.6% to 51.8%) | **51.5%** | **0.300** |
| Exact identifier | 20.5% | 30.0% | 38.9% | 48.3% | 0.271 |
| Patients whose disease keeps three or more annotations (77%) | 25.4% | 40.3% | 50.0% | 62.7% | 0.349 |
| Causal gene, ranked by its best disease | 31.5% | 41.5% | 50.7% | 61.6% | 0.384 |

The probability of the leading candidate is well calibrated: expected calibration error 0.046, Brier score 0.134.

<p align="center">
<img width="800" height="736" alt="calibration" src="https://github.com/user-attachments/assets/f26c541b-c346-4ddd-9a98-76f522bd071e" />
</p>

### The leakage that the strict protocol removes

<p align="center">
<img width="1200" height="640" alt="leakage" src="https://github.com/user-attachments/assets/88d69628-7865-40ed-ab0d-d49ca71996d6" />
</p>


| Same 2,045 blind patients, Mondo concept | Top 1 | Top 10 | MRR |
| --- | --- | --- | --- |
| **Strict: the patient's own paper removed (reported)** | **22.7%** | **41.7%** | **0.300** |
| Released annotations, as most benchmarks score | 56.0% | 75.2% | 0.635 |

Across all eligible patients:

* 74.8% come from a paper that the disease annotations also cite;
* on average 34.5% of the true disease's annotations rest on that paper alone;
* for 19.7% of patients, every annotation does.

Scoring against the released annotations makes the same system look two and a half times better at the first position. Reporting it would mean claiming credit for remembering the patient's own description.

### Rankers

<p align="center">
<img width="1760" height="704" alt="rankers" src="https://github.com/user-attachments/assets/9fc63857-a4ad-4a4c-876c-5e4277940633" />
</p>

| Exact identifier, blind | Top 1 | Top 10 | Top 50 | MRR |
| --- | --- | --- | --- | --- |
| **Fused ranker (product)** | **20.5%** | **38.9%** | **48.3%** | **0.271** |
| Likelihood ratio | 18.3% | 34.3% | 43.5% | 0.238 |
| Resnik, patient to disease | 19.9% | 37.2% | 46.1% | 0.259 |
| Exact term overlap | 18.6% | 36.1% | 44.3% | 0.255 |
| BM25 over disease text | 14.4% | 37.6% | 52.1% | 0.220 |
| Resnik best match average | 10.7% | 33.4% | 46.9% | 0.180 |
| Latent semantic over disease text | 5.7% | 24.7% | 41.6% | 0.120 |

The browser ranker, which uses ontology features only, reaches 23.0% top 1 and 42.0% top 10 at concept level, level with the product.

<p align="center">
<img width="1200" height="672" alt="topk_curve" src="https://github.com/user-attachments/assets/226f88f6-43b0-46c4-b0ce-18c50c741a6a" />
</p>

### A curation artifact, found and removed before the blind run

The first development run showed the next best question making things worse: answering "absent" for a feature the true disease almost never has lowered that disease's probability 91% of the time.

The cause is in how published cases are written. A cohort paper records which of its disease's typical features each patient lacked, so in the development data the true disease shows 1.45 of the patient's excluded features on average, against 1.01 for its competitors. The unconstrained ranker had learned that "a typical feature of this disease is absent" is evidence for the disease. That is true of journals and false in a clinic.

Both learned rankers are now constrained so that every piece of evidence can only push a disease's score in its clinical direction. This costs four points of top 1 in development (27.2% against 31.4%), the measured value of the artifact. In exchange, an absent answer now raises the true disease's probability in 43% of blind cases and lowers it in 15%.

### Next best question

<p align="center">
<img width="1920" height="672" alt="next_question" src="https://github.com/user-attachments/assets/1c8fc061-a75f-4554-a28f-57d3048f4dfc" />
</p>

| 200 blind patients, half the features hidden, five questions | Top 1 start | Top 1 after 5 | MRR after 5 |
| --- | --- | --- | --- |
| **Expected information gain (product)** | 18.5% | **27.0%** | **0.325** |
| Most frequent feature of the leading candidate | 18.5% | 22.0% | 0.301 |
| Random askable feature | 18.5% | 22.5% | 0.295 |

Answers come from a simulated patient with the true disease, drawn from its annotation frequencies. When only the published record may answer, fewer than one question in five is documented, and every strategy moves the ranking by less than two points.

### Knowledge retrieval

<p align="center">
<img width="1200" height="576" alt="retrieval" src="https://github.com/user-attachments/assets/d2e676b1-442d-403f-ac89-71a600607013" />
</p>

| 403 blind questions | Hit@1 | Recall@3 | MRR |
| --- | --- | --- | --- |
| **Learned graph and text ranker (product)** | **0.923** | **1.000** | **0.961** |
| BM25 | 0.400 | 0.521 | 0.470 |
| Hybrid BM25 and latent semantic, equal weights | 0.382 | 0.491 | 0.451 |
| Latent semantic | 0.107 | 0.159 | 0.157 |

30 of the 31 misses retrieved the right disease under its other identifier (OMIM instead of Orphanet, or the reverse). Every in session question ("how is the leading candidate inherited") was answered from the right passage.

### Extraction, copilot and safety

| Component | Blind result |
| --- | --- |
| Present features read from rendered notes | precision 93.4%, recall 99.0% |
| Absent features | precision 98.6%, recall 88.1% |
| Family history read as the patient's | 0 of 4,090 distractors |
| Note pathway end to end (rendered note, extraction, ranking) | top 10 43.1% against 41.7% from structured terms |
| Composed sentences that passed the verifier | 4,271 of 4,487 (95.2%); 3,261 of 3,262 after two composer fixes, see below |
| Verifier catching changed identifiers, wrong citations, treatment, certainty, genes, doses | 99.0% to 100% |
| Safety probes | 25 of 25 |
| Median copilot run | 459 ms on two cores |

<p align="center">
<img width="1200" height="608" alt="verifier" src="https://github.com/user-attachments/assets/5f307b9b-b8b0-4d5e-a943-dd2c475f4d6c" />
</p>

### Where it fails

| Failure | Evidence |
| --- | --- |
| Diseases known only from the patient's paper | 22.8% of blind patients have no annotation left after strict removal; 13.7% of them reach the top 10, through Orphanet records and definitions |
| The fused ranker adds little on unseen diseases | identifier top 10 of 38.9% against 37.6% for BM25 and 37.2% for Resnik similarity; the ontology only browser ranker is level with it |
| A changed number often survives the verifier | 69.3% caught, because the model output lists many numbers and a changed one can coincide with another |
| Two composer defects reached the blind run | the opening sentence stated counts with no source and contradictions printed raw frequency codes; the verifier dropped those sentences (216 of 4,487) rather than show them; both were fixed afterwards and 3,261 of 3,262 sentences now verify on development cases |
| Atypical inheritance | 9.5% top 10 for diseases whose inheritance is mixed or unusual |
| Sparse or vague descriptions | 24.6% top 10 with three features, 28.6% when every feature is one level too general |
| The ranker leans on profile size | 43% of split gain; the strict protocol shrinks the true disease's profile, so part of this may not transfer to unstripped use |
| Extraction is measured on generated prose | notes are rendered from structured cases, not written by clinicians |


<p align="center">
<img width="1200" height="576" alt="robustness" src="https://github.com/user-attachments/assets/3b2b1847-8235-4304-b0e7-273dd3480f9f" />
</p>

<p align="center">
<img width="1200" height="1904" alt="subgroups" src="https://github.com/user-attachments/assets/c7fa9f74-8aa6-43fe-95f6-c11417b38ebb" />
</p>

## Architecture

<p align="center">
<img width="2400" height="1770" alt="architecture" src="https://github.com/user-attachments/assets/b90eca54-629e-4d05-a0a6-84aad8050cd8" />
</p>


```
GitHub release assets ──► raw sources and SHA256 manifest                     make data
        ▼
parse, normalise, strip treatment text ──► processed tables
        ▼
freeze and hash development and blind sets                                    make freeze
        ▼
ontology graph, disease profiles, strict index, 112,371 passages, text indexes make index
        ▼
likelihood ratio settings, strict features, constrained rankers, calibration   make tune features train
        ▼
copilot: guard ► extract ► normalise ► score ► fuse ► group ► explain ► discriminate ► genes ► retrieve ► compose ► verify ► audit
        ▼
development evaluation, then one blind run                                     make evaluate, make evaluate_blind
        ▼
FastAPI service and single page interface; the interface also runs alone in the browser
```

Two retrieval worlds share one set of identifiers. The ontology graph ranks diseases from structured phenotypes; the passage store answers questions in words. A disease, phenotype or gene has the same identifier in both, so any sentence can cite either and the verifier checks both the same way. See [docs/architecture.md](docs/architecture.md).

## Data foundation

| Source | Rows | Used for |
| --- | --- | --- |
| Human Phenotype Ontology (2026-09-01) | 19,894 current terms, 24,436 is_a edges | phenotype vocabulary, hierarchy, synonyms, definitions, information content |
| HPO disease annotations (2026-09-02) | 286,651 rows over 12,880 OMIM, Orphanet and DECIPHER diseases | disease profiles with frequency, NOT annotations and references |
| Mondo Disease Ontology (2026-10-06) | 32,136 classes | disease identity, OMIM and Orphanet merging, definitions with treatment removed |
| HPO gene links | 12,673 gene to disease links | gene panels and gene passages |
| GA4GH phenopacket store (0.1.27) | 10,377 published patients, 780 diseases | evaluation only: 7,089 development and 2,045 blind patients |
| Knowledge base | 112,371 passages, 4.3 million tokens | disease overviews, disease by organ system, phenotype terms, genes, disease groups |
| Questions | 550 frozen, plus in session versions | retrieval training (development) and evaluation (blind) |

Every file is fingerprinted in `data/raw/MANIFEST.json` and every table in `data/processed/MANIFEST.json`. See [docs/data_card.md](docs/data_card.md).

## Methods

| Area | Method |
| --- | --- |
| Ingestion | HPO, annotations, gene links and Mondo parsed to flat tables; obsolete and alternative identifiers mapped; frequency terms to band midpoints and counts smoothed as (n + 0.5) / (m + 1); every treatment sentence removed from definitions |
| Protocol | cohorts joined into groups when they share a disease; a quarter of the groups blind; five grouped development folds; blind cases and questions hashed before anything was built |
| Strict scoring | for each patient, every annotation whose only reference is the patient's paper is removed and the affected diseases are rescored, text documents included |
| Ontology | ancestor closure as a sparse matrix; annotation based information content; background frequency as the mean propagated frequency |
| Likelihood ratio | for each observed term the chance a patient with the disease shows it, by the best of three routes (the same or a more specific annotation; a more general annotation scaled by the background ratio; a shared ancestor, down weighted), smoothed to the background and divided by it; a NOT annotation that covers the term caps the ratio at a floor; each excluded term contributes (1 minus 0.95 times its frequency) over (1 minus its background); settings chosen on development cases |
| Semantic similarity | Resnik best match average in both directions, the disease direction weighted by annotation frequency |
| Text rankers | BM25 and a 192 dimension latent semantic space over one document per disease (names, synonyms, definition, features) |
| Fused ranker | shortlist from every base ranker; 25 features (raw signals, gaps to the best, log ranks on values rounded to three decimals); every feature constrained to push a disease's score in its clinical direction only; LambdaMART against a sign bounded linear pairwise ranker under a preregistered rule; temperature fitted on out of fold scores |
| Concepts | OMIM and Orphanet records merged by their Mondo class; concept probability is the sum over records |
| Evidence | for every patient term, the annotation with the most informative shared ancestor, classed as exact, more specific, more general, related or unexplained; contradictions; characteristic features not yet recorded, deduplicated by ontology relation |
| Next question | expected information gain over the posterior of the leading candidates, using propagated frequencies smoothed to the background |
| Extraction | longest match over 43,062 normalised phrases with order insensitive matching for short phrases; clause scoped negation before and after the mention; family history with self reference reset; uncertainty cues |
| Knowledge retrieval | BM25 and a 256 dimension latent semantic space over 112,371 passages, plus a graph route through linked diseases, genes and phenotypes and the current differential; an eleven feature logistic ranker trained on development questions and their in session variants |
| Composition | deterministic cited sentences; answers taken from the first passage whose kind fits the question's intent; an optional language model writer held to the same verifier |
| Verifier | citations exist; at least 40% of content words, every number, identifier and gene symbol present in the cited text; no treatment or dose; no claim of certainty |
| Audit | append only log of redacted runs, each record chained to the previous by SHA256 |
| Browser | JavaScript port of the recogniser, both scorers, the ontology only ranker, the explainer and the next question, checked against Python on 300 cases and 150 notes |

## Running it

Requirements: Python 3.11 or later, Node 20 or later for the browser tests, about 2 GB of free disk and 4 GB of memory.

```
pip install ".[dev]"
make data          # download the five public sources and build the tables
make freeze        # freeze and hash the evaluation sets (already done; ships with the repository)
make index         # knowledge base passages, text indexes, passage ranker
make tune features train
make evaluate      # development evaluation
make test          # Python unit and integration tests, then the JavaScript parity tests
make serve         # API and interface on http://localhost:8000
```

The frozen sets, the blind run log, the trained models, the reports and the built interface ship with the repository, so `make data index` followed by `make serve` is enough to use the system. `make evaluate_blind` scores the blind set again and appends a timestamp to `data/eval/blind_runs.json`, so any later run is visible. With Docker, `docker compose up --build` serves the API and interface on port 8000. The optional language model writer and static hosting are described in [docs/deployment.md](docs/deployment.md), and the API in [docs/api.md](docs/api.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `odyssey_anchor/ingest` | download with fingerprints, parsers for HPO, annotations, Mondo, genes and phenopackets, table build |
| `odyssey_anchor/ontology` | the ontology graph, disease profiles, information content, the strict index |
| `odyssey_anchor/knowledge` | passage identifiers and the knowledge base build |
| `odyssey_anchor/retrieval` | tokeniser and lexicon matcher, BM25, latent semantic space, entity linker, passage ranker |
| `odyssey_anchor/extract` | phenotype recogniser and the note renderer used for testing |
| `odyssey_anchor/rank` | scorers, engine, features, setting search, development features, fused rankers |
| `odyssey_anchor/reason` | evidence subgraphs, next best question, gene panel |
| `odyssey_anchor/agent` | guard, composer, verifier, optional language model writer, audit chain, copilot state machine |
| `odyssey_anchor/eval` | freeze, runner, metrics, perturbations, extraction, retrieval, verifier and next question evaluations, probes, leakage, figures |
| `odyssey_anchor/export` | browser bundle and parity fixture, teaching cases, screenshots |
| `odyssey_anchor/api` | FastAPI service |
| `web` | interface source, browser engine, parity tests, build script, built pages |
| `data` | raw manifest, processed tables, frozen evaluation sets, knowledge base, models, audit logs |
| `reports` | evaluation reports, blind predictions, teaching cases, figures |
| `docs` | architecture, data card, model card, evaluation, safety, API, deployment, diagrams, screenshots |
| `tests` | unit tests on a synthetic ontology and integration tests on the real knowledge base |

## Intended use and limitations

Odyssey Anchor AI is a research and teaching tool. It shows how a phenotype ranker, an evidence explainer and a cited, verified retrieval copilot could support a clinical geneticist working through a rare disease differential, and how much an honest evaluation changes the numbers. It is not a medical device and must not be used to make or delay decisions about real patients.

1. The patients are real but published: journal descriptions curated into phenopackets, not clinic notes. Their excluded features carry a curation artifact that the rankers are constrained against but cannot be fully free of.
2. The strict protocol removes annotations resting only on the patient's paper. Annotations backed by an OMIM clinical synopsis are kept, even though the synopsis may summarise that paper.
3. Probabilities are calibrated on published cases and are relative to the shortlist; they are not the probability that a real patient has the disease.
4. The note pathway is tested on notes rendered from structured cases. Real clinical prose is more varied.
5. The verifier checks words, numbers and identifiers, not meaning, and lets about three in ten changed numbers through.
6. The knowledge base restates public ontologies at one release; it is not reviewed by a clinical geneticist and contains no treatment information by design.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | layers, data flow, the knowledge layer, the thirteen node runtime path, the browser engine, make targets |
| [docs/data_card.md](docs/data_card.md) | every source and release, every table, transformations, patients, splits, known issues |
| [docs/model_card.md](docs/model_card.md) | every learned or tuned component, its data, results and failure modes |
| [docs/evaluation.md](docs/evaluation.md) | the protocol, every blind result, failures and the roadmap |
| [docs/safety.md](docs/safety.md) | what the system never does, the guard, the verifier, the audit chain, known gaps |
| [docs/api.md](docs/api.md) | endpoints, request and response fields |
| [docs/deployment.md](docs/deployment.md) | local, Docker, the optional writer, static hosting, updating the data |

## Licence

The code is released under the MIT licence; see [LICENSE](LICENSE). The Human Phenotype Ontology and its annotations are used under the HPO terms of use and are downloaded, not redistributed; Mondo and the phenopacket store are CC BY 4.0. Passages restate these sources with their identifiers and references so that every statement can be traced back.

## References

1. Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes around the world. Nucleic Acids Research 2024.
2. Vasilevsky NA, Matentzoglu NA, Toro S, et al. Mondo: unifying diseases for the world, by the world. medRxiv 2022.
3. Danis D, Bamshad MJ, Bridges Y, et al. A corpus of GA4GH phenopackets: case level phenotyping for genomic diagnostics and discovery. HGG Advances 2025.
4. Jacobsen JOB, Baudis M, Baynam GS, et al. The GA4GH Phenopacket schema defines a computable representation of clinical data. Nature Biotechnology 2022.
5. Köhler S, Schulz MH, Krawitz P, et al. Clinical diagnostics in human genetics with semantic similarity searches in ontologies. American Journal of Human Genetics 2009.
6. Robinson PN, Ravanmehr V, Jacobsen JOB, et al. Interpretable clinical genomics with a likelihood ratio paradigm. American Journal of Human Genetics 2020.
7. Resnik P. Using information content to evaluate semantic similarity in a taxonomy. IJCAI 1995.
8. Burges CJC. From RankNet to LambdaRank to LambdaMART: an overview. Microsoft Research technical report 2010.
9. Robertson S, Zaragoza H. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval 2009.
10. Chapman WW, Bridewell W, Hanbury P, et al. A simple algorithm for identifying negated findings and diseases in discharge summaries. Journal of Biomedical Informatics 2001.
11. Smedley D, Jacobsen JOB, Jäger M, et al. Next generation diagnostics and disease gene discovery with the Exomiser. Nature Protocols 2015.
12. Nguengang Wakap S, Lambert DM, Olry A, et al. Estimating cumulative point prevalence of rare diseases: analysis of the Orphanet database. European Journal of Human Genetics 2020.
