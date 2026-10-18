# Evaluation

## Protocol

The protocol was designed before anything was built, and the order of work enforced it.

1. **Freeze first.** The processed patient table was split before any index, model or setting existed. Cohorts (one per gene or chromosomal disorder in the phenopacket store) were joined into groups whenever they shared a disease, and a quarter of the groups, chosen with a fixed seed, became the blind set. The blind cases and blind questions were hashed into `data/eval/FROZEN.json` at 19:56 UTC on 17 October 2026.
2. **Develop on development cases only.** Likelihood ratio settings, ranker features, model choice, temperatures and the passage ranker were all chosen on the 7,089 development patients and 295 development questions, with five folds grouped by cohort group. The model choice rule was written before cross validation ran.
3. **Score strictly.** For every patient, every annotation whose only reference is that patient's paper is removed and the affected diseases are rescored, their text documents included. Annotations that also cite another paper or OMIM are kept.
4. **Score the blind set once.** `make evaluate_blind` recomputes both hashes, refuses to run if either changed, and appends a timestamped line to `data/eval/blind_runs.json`. Teaching cases and figures are drawn from the stored blind predictions, without rescoring.
5. **Split ties fairly.** Every rank is the mid rank of its tie: a disease tied with nine others for first place counts as rank 5.5, not 1. Coarse scores such as exact overlap cannot win by tying.

Accuracy is reported at two levels:

* **Mondo concept** (headline): an OMIM and an Orphanet record of the same disease count as the same answer, which is what a clinician sees.
* **Exact identifier**: only the OMIM record the paper reported counts.

Confidence intervals resample cohort groups, not patients, because patients in one cohort share a disease and a paper.

The blind set holds 2,045 patients in 170 cohort groups and 188 diseases. It was scored once, from 23:28:31 to 00:33:02 UTC on 17 to 18 October 2026, and `data/eval/blind_runs.json` holds that single entry.

## Headline

| Level | Top 1 | Top 3 | Top 10 | Top 50 | MRR | Median rank |
| --- | --- | --- | --- | --- | --- | --- |
| Mondo concept | 22.7% (16.4% to 29.3%) | 34.2% | 41.7% (31.6% to 51.8%) | 51.5% | 0.300 (0.231 to 0.374) | 35 |
| Exact identifier | 20.5% | 30.0% | 38.9% | 48.3% | 0.271 | 64 |

The true disease reached the fused ranker's shortlist for 61.8% of patients. For the 77% whose disease keeps at least three annotations after strict removal, the concept top 1 is 25.4% and the top 10 is 50.0%.

## Leakage

![leakage](../reports/figures/leakage.png)

| Same blind patients, Mondo concept | Top 1 | Top 3 | Top 10 | Top 50 | MRR |
| --- | --- | --- | --- | --- | --- |
| Strict protocol (reported) | 22.7% | 34.2% | 41.7% | 51.5% | 0.300 |
| Released annotations | 56.0% | 69.0% | 75.2% | 81.6% | 0.635 |

Across all 9,134 eligible patients:

* 74.8% come from a paper the annotations cite;
* on average 34.5% of the true disease's annotations rest on that paper alone;
* for 19.7% of patients, every annotation of the true disease does.

Scored against the released annotations, the same system looks two and a half times better at the first position.

## Rankers

![rankers](../reports/figures/rankers.png)

| Exact identifier, blind | Top 1 | Top 10 | Top 50 | MRR |
| --- | --- | --- | --- | --- |
| **Fused ranker (product)** | **20.5%** | **38.9%** | **48.3%** | **0.271** |
| Likelihood ratio | 18.3% | 34.3% | 43.5% | 0.238 |
| Resnik best match average | 10.7% | 33.4% | 46.9% | 0.180 |
| Resnik, patient to disease | 19.9% | 37.2% | 46.1% | 0.259 |
| Resnik, disease to patient | 0.3% | 3.5% | 9.8% | 0.015 |
| Exact term overlap | 18.6% | 36.1% | 44.3% | 0.255 |
| BM25 over disease text | 14.4% | 37.6% | 52.1% | 0.220 |
| Latent semantic over disease text | 5.7% | 24.7% | 41.6% | 0.120 |

The browser ranker (ontology features only) reaches 23.0% top 1 and 42.0% top 10 at concept level, level with the product.

![cumulative accuracy](../reports/figures/topk_curve.png)

## Genes

For the 1,976 blind patients with a causative gene, ranking each gene by its best disease puts the gene first for 31.5% and in the top 10 for 50.7% (MRR 0.384, median rank 9).

## Calibration

![calibration](../reports/figures/calibration.png)

The probability of the leading concept has an expected calibration error of 0.046 and a Brier score of 0.134 on the blind set.

## Subgroups

![subgroups](../reports/figures/subgroups.png)

| Group | Patients | Top 1 | Top 10 | MRR |
| --- | --- | --- | --- | --- |
| Female | 901 | 20.1% | 38.3% | 0.274 |
| Male | 985 | 22.7% | 41.8% | 0.299 |
| Sex not recorded | 159 | 37.1% | 59.7% | 0.459 |
| Under 1 year | 171 | 24.0% | 48.0% | 0.335 |
| 1 to 5 years | 408 | 26.5% | 43.6% | 0.331 |
| 6 to 17 years | 544 | 19.1% | 37.9% | 0.269 |
| 18 and over | 445 | 24.0% | 47.4% | 0.324 |
| Age not recorded | 477 | 21.8% | 36.7% | 0.275 |
| 3 to 5 observed features | 578 | 16.3% | 41.2% | 0.258 |
| 6 to 10 | 764 | 23.4% | 41.6% | 0.310 |
| 11 to 20 | 544 | 26.7% | 42.8% | 0.321 |
| Over 20 | 159 | 28.9% | 39.6% | 0.335 |
| No annotations left after strict removal | 466 | 13.7% | 13.7% | 0.137 |
| 1 to 9 left | 105 | 6.7% | 26.7% | 0.146 |
| 10 to 29 left | 386 | 19.7% | 40.7% | 0.269 |
| 30 or more left | 1,088 | 29.1% | 55.4% | 0.396 |
| Autosomal dominant | 901 | 22.1% | 46.4% | 0.321 |
| Autosomal recessive | 866 | 27.1% | 43.8% | 0.330 |
| X linked | 137 | 19.7% | 29.2% | 0.228 |
| Mixed or other | 137 | 2.2% | 9.5% | 0.050 |

The number of annotations left matters far more than any patient characteristic. Female and male patients differ by 3.5 points at the top 10, within the cohort level uncertainty.

## Robustness

![robustness](../reports/figures/robustness.png)

| Every blind patient, changed | Top 1 | Top 10 | MRR |
| --- | --- | --- | --- |
| As published | 22.7% | 41.7% | 0.300 |
| Half the observed features dropped | 13.9% | 27.1% | 0.186 |
| Only three observed features | 10.8% | 24.6% | 0.157 |
| Excluded features removed | 23.9% | 42.7% | 0.310 |
| Every feature replaced by a parent term | 12.6% | 28.6% | 0.184 |
| Two unrelated common features added | 20.6% | 38.7% | 0.275 |
| Rendered as a note, then extracted | 24.9% | 43.1% | 0.321 |

Removing excluded features helps slightly, and so does the note pathway, which renders at most eight of them. Even with constrained rankers, published excluded lists carry some of the curation artifact described in the model card. Two unrelated features cost about three points at the top 10.

## Next best question

![next question](../reports/figures/next_question.png)

200 blind patients; half of each patient's observed features hidden, five questions asked.

| Strategy | Answers | MRR start | MRR after 5 | Top 1 start | Top 1 after 5 |
| --- | --- | --- | --- | --- | --- |
| Information gain (product), disease model oracle | 5.0 of 5 | 0.262 | 0.325 | 18.5% | 27.0% |
| Most frequent feature of the leader, disease model oracle | 4.85 | 0.262 | 0.301 | 18.5% | 22.0% |
| Random askable feature, disease model oracle | 5.0 | 0.262 | 0.295 | 18.5% | 22.5% |
| Information gain, record oracle | 0.69 | 0.262 | 0.276 | 18.5% | 20.0% |
| Most frequent, record oracle | 0.54 | 0.262 | 0.270 | 18.5% | 19.5% |
| Random, record oracle | 0.48 | 0.262 | 0.273 | 18.5% | 20.5% |

The behavioural check on the information gain questions:

* an answer of present raised the true disease's probability 64% of the time and lowered it 15% of the time;
* an answer of absent raised it 43% of the time and lowered it 15% of the time.

Before the rankers were constrained, an absent answer lowered it 91% of the time on development cases.

## Extraction

2,045 blind patients rendered as notes.

| | Precision | Recall |
| --- | --- | --- |
| Present features | 93.4% | 99.0% |
| Absent features | 98.6% | 88.1% |
| Present, ancestor tolerant | 93.6% | |
| Family history read as the patient's | 0 of 4,090 | |

## Knowledge retrieval

![retrieval](../reports/figures/retrieval.png)

403 blind questions: 255 frozen ones plus an in session version of each disease question.

| Method | Hit@1 | Recall@3 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- |
| BM25 | 0.400 | 0.521 | 0.470 | 0.487 |
| Latent semantic | 0.107 | 0.159 | 0.157 | 0.169 |
| Hybrid, equal weights | 0.382 | 0.491 | 0.451 | 0.472 |
| **Learned ranker (product)** | **0.923** | **1.000** | **0.961** | **0.971** |
| Learned, without intent fit | 0.861 | 0.990 | 0.923 | 0.943 |
| Learned, without the disease link | 0.831 | 0.931 | 0.882 | 0.897 |
| Learned, without the semantic score | 0.923 | 1.000 | 0.961 | 0.971 |

By kind (hit@1), with the number of questions in brackets:

| Kind | Hit@1 |
| --- | --- |
| Definitions | 1.000 (55) |
| Diseases of a gene | 1.000 (52) |
| Feature in a disease | 0.829 (41) |
| Genes of a disease | 0.741 (54) |
| Inheritance | 0.811 (53) |
| Every in session version | 1.000 |

30 of the 31 misses retrieved the right disease under its other identifier (the Orphanet record instead of the OMIM one, or the reverse); counted at the Mondo concept level, as the differential is, hit@1 would be 0.998.

## Copilot and verifier

![verifier](../reports/figures/verifier.png)

Over 200 blind copilot runs, 4,271 of 4,487 composed sentences passed the verifier (95.2%). The median run took 459 ms on two cores.

The 216 dropped sentences break down as follows:

* 115 failed on identifiers and 95 on numbers. Both were composer defects that the verifier caught: frequency codes printed raw, and counts stated without a source. They were fixed after the blind run. On 150 development cases, 3,261 of 3,262 sentences now verify (`reports/copilot_postfix_dev.json`).
* 6 were treatment sentences, rightly dropped.

| Corruption of a verified sentence | Sentences | Caught |
| --- | --- | --- |
| A number changed | 1,954 | 69.3% |
| A disease identifier swapped | 1,000 | 99.0% |
| Citations removed | 4,071 | 100% |
| A citation pointed at another passage | 3,966 | 99.4% |
| Treatment advice added | 4,071 | 100% |
| A definitive diagnosis added | 4,071 | 100% |
| A gene symbol swapped | 282 | 100% |
| A dose added | 4,071 | 100% |

Unaltered control sentences pass 100% of the time. All 25 safety probes passed.

## Where it fails

| Failure | Evidence |
| --- | --- |
| Diseases known only from the patient's paper | 22.8% of blind patients have no annotation left after strict removal; 13.7% of them reach the top 10 |
| The fused ranker adds little over its inputs on new diseases | identifier top 10 is 38.9% against 37.6% for BM25 and 37.2% for Resnik patient to disease similarity; the ontology only browser ranker is level |
| A changed number often survives the verifier | 69.3% caught: the model output source lists many numbers, so a changed one can coincide with another |
| Atypical inheritance | 9.5% top 10 for diseases with mixed or other inheritance |
| Sparse or vague input | 24.6% top 10 with three features; 28.6% when every feature is one level more general |
| Answers from published records | a published case documents fewer than one of five suggested features, so its record cannot show the question's value |
| Rendered notes are not clinic notes | extraction is measured on prose generated from structured cases |

## Roadmap

1. Score candidates with the Orphanet record and its OMIM twins together, so diseases known only from one paper can still be matched through the group record.
2. Give the verifier the number's role: a probability must match the probability of the named disease, not any number in the source.
3. Add variant evidence (a VCF) and rank genes jointly with phenotype, as variant prioritisation tools do.
4. Annotate a set of real, consented clinic notes to measure extraction where it matters.
5. Learn the frequency of absent findings from clinic data, not published cases, to remove the remaining curation artifact.
