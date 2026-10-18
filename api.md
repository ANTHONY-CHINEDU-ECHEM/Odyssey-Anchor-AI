# API

Start the service with `make serve` (or `docker compose up`). It listens on port 8000, serves the interface at `/` and the interactive OpenAPI documentation at `/docs`.

## `POST /v1/analyse`

A clinical note, structured terms, or both, with an optional question.

```json
{
  "note": "Tall 14 year old boy with arachnodactyly, ectopia lentis and aortic root aneurysm. No intellectual disability.",
  "observed": ["HP:0001166"],
  "excluded": [],
  "question": "How is the leading candidate inherited?",
  "sex": "MALE",
  "age_years": 14,
  "strict_pmid": null
}
```

`strict_pmid` is for published cases: annotations resting only on that paper are removed before scoring.

The response contains:

| Field | Content |
| --- | --- |
| `run_id` | identifier of the run, also in the audit log |
| `guard` | flags, redaction counts and fixed messages |
| `findings` | every recognised phenotype with span, status (present, absent, family, uncertain), cue and how it matched |
| `patient` | the terms used: observed (most specific), excluded, uncertain, family |
| `differential` | up to ten Mondo concepts: rank, name, representative disease, linked records, probability, genes, inheritance, base ranks and an `evidence` object (matches with ontology links, contradictions, features to check) |
| `questions` | next best questions with expected gain in bits and the candidate favoured by each answer |
| `genes` | gene panel with weights and the diseases behind each gene |
| `answer` | the question's intent, linked entities, retrieved passages and which writer answered |
| `sentences` | verified sentences with their citations and role |
| `verification` | passed and total counts, and the dropped sentences with reasons |
| `sources` | the text of every cited source |
| `timings_ms` | milliseconds per node |
| `audit_hash` | the hash of the audit record |

## `POST /v1/ask`

```json
{ "question": "What conditions are caused by variants in FBN1?", "context": [] }
```

`context` takes disease identifiers from the current differential, so a question can refer to "the leading candidate". The response has `answer`, `sentences`, `verification`, `sources` and `timings_ms`.

## `POST /v1/extract`

```json
{ "note": "No seizures but marked hypotonia. Her mother has myopia." }
```

Returns the findings with spans and the term lists, without ranking.

## Lookups

| Endpoint | Returns |
| --- | --- |
| `GET /v1/terms/search?q=ectop` | phenotype terms by label or synonym |
| `GET /v1/diseases/OMIM:154700` | name, genes, inheritance and every passage of a disease |
| `GET /v1/passages/{pid}` | one passage by identifier, for example `HPO|HP:0001083` |
| `GET /v1/cases` | the teaching cases shown in the interface |
| `GET /v1/reports/{evaluation_blind,evaluation_dev,lr_tuning}` | evaluation reports |
| `GET /v1/audit/verify` | `{"valid": true, "records": n, "head": "..."}` or the index of the first broken record |
| `GET /health` | status, version and data releases |

## Errors

| Status | When |
| --- | --- |
| 400 | `/v1/analyse` without a note, terms or question |
| 404 | unknown disease, passage or report |
| 422 | a body that does not match the schema, for example an age over 120 |
