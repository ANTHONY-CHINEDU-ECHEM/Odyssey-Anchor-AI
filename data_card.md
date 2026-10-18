# Data card

Every input is public, versioned and downloaded by `make data` from GitHub release assets. The exact files used are fingerprinted in `data/raw/MANIFEST.json`, and the processed tables in `data/processed/MANIFEST.json`.

## Sources

| Source | Release used | Licence | Role |
| --- | --- | --- | --- |
| Human Phenotype Ontology, `hp.json` | 2026-09-01 | HPO terms of use, attribution required | Phenotype vocabulary, synonyms, definitions, hierarchy |
| HPO annotations, `phenotype.hpoa` | 2026-09-02 | HPO terms of use | Disease to phenotype links with frequency, onset, sex, NOT qualifier and references |
| HPO gene links, `genes_to_phenotype.txt` | same release | HPO terms of use | Gene to disease links (only the gene and disease columns are used) |
| Mondo Disease Ontology, `mondo.json` | 2026-10-06 | CC BY 4.0 | Disease identity, OMIM and Orphanet cross references, definitions, hierarchy |
| GA4GH phenopacket store, `all_phenopackets.zip` | 0.1.27 | CC BY 4.0 | Real published patients, used only to evaluate |

## Tables

| Table | Rows | Notes |
| --- | --- | --- |
| `hpo_terms` | 20,482 (19,894 current) | label, definition, synonyms with type, cross references, obsolete flag, replacement, organ system |
| `hpo_edges` | 24,436 | is_a between current terms |
| `hpo_alt` | 3,975 | alternative and obsolete identifiers mapped to current ones |
| `annotations` | 286,651 | 8,478 OMIM, 4,355 Orphanet and 47 DECIPHER diseases; aspects P (phenotype), I (inheritance), C (clinical course), M and H |
| `mondo_classes` | 32,136 | non obsolete Mondo classes with treatment free definitions |
| `mondo_edges`, `mondo_xrefs` | 46,732 and 19,132 | hierarchy, and cross references normalised to `OMIM:` and `ORPHA:` |
| `disease_genes` | 12,673 | gene symbol, NCBI gene identifier, disease |
| `diseases` | 12,880 | one row per annotated disease with Mondo identity, definition, synonyms, inheritance, onset, genes and counts |
| `cases` | 10,377 | one row per published patient |

## Transformations

* **Frequencies.** The HPO frequency terms map to the middle of their bands: obligate 1.0, very frequent 0.895, frequent 0.545, occasional 0.17, very rare 0.025, excluded 0. Counts such as 7/12 are smoothed as (n + 0.5) / (m + 1), so 1/1 becomes 0.75 rather than certainty. Percentages are used as stated, and a missing frequency becomes 0.5.
* **Identifiers.** Alternative and obsolete HPO identifiers are mapped to current ones, both in annotations and in patients. Every OMIM and Orphanet cross reference in this Mondo release points to exactly one Mondo class.
* **Treatment.** Every sentence in a Mondo definition that mentions treatment, therapy, dosing, drugs, surgery, management, prognosis or cure is removed (1,021 sentences). The system supports diagnosis only, so this text never enters the knowledge base.
* **Organ system.** Each phenotype term is assigned to one child of *Phenotypic abnormality*; when it sits under several, the lowest identifier is used, consistently everywhere.

## Patients

The phenopacket store collects individuals described in peer reviewed papers, each with observed and explicitly excluded HPO features, a confirmed diagnosis, the causative variant and the source PubMed identifier.

| Property | Value |
| --- | --- |
| Patients | 10,377 in 700 cohorts (one per gene or chromosomal disorder) |
| Diseases | 780, all OMIM |
| Observed features | median 7, up to 66 |
| Excluded features | median 7, up to 174 |
| Sex | 5,060 male, 4,517 female, 800 not recorded |
| Age at last encounter | recorded for 77% |
| Eligible for evaluation | 9,134: diagnosis in the knowledge base and at least three observed features |

## Splits

Cohorts are joined into groups whenever they share a disease, so no disease appears in both splits. A quarter of the groups, chosen with a fixed seed, form the blind set:

| Split | Patients | Groups | Diseases |
| --- | --- | --- | --- |
| Development | 7,089 | 508 | 568 |
| Blind | 2,045 | 170 | 188 |

Development groups are assigned round robin to five folds for grouped cross validation. The blind set was hashed at 19:56 UTC on 17 October 2026, before any index, model or setting existed.

## Questions

550 knowledge questions (295 development, 255 blind) are generated from the processed tables in five kinds: genes of a disease, inheritance of a disease, meaning of a phenotype, whether a feature is part of a disease, and diseases caused by a gene. Disease and phenotype names are written through synonyms rather than the canonical label, so retrieval cannot rely on exact title matches. Evaluation also asks an in session version of each disease question ("how is the leading candidate inherited") with the differential as context.

## Known issues

* **Leakage.** 74.8% of eligible patients come from a paper that the annotations also cite. On average 34.5% of the true disease's annotations rest on that paper alone, and for 19.7% of patients every annotation does. This is why every reported number uses the strict protocol (see [evaluation.md](evaluation.md)).
* **Residual leakage.** Annotations backed by an OMIM clinical synopsis are kept, even though the synopsis may itself summarise the patient's paper.
* **Research descriptions.** Published cases are written for journals, not clinics. Their excluded features are often long lists of normal findings, which are rarely recorded in practice.
* **Coverage.** The 780 diseases are those with a curated cohort, mostly monogenic disorders with a known gene. Chromosomal and multifactorial conditions are under represented.
