# Safety

Odyssey Anchor AI is decision support for clinicians working through a rare disease differential. Its safety design rests on three rules: it never names a diagnosis as fact, it never advises on treatment, and it never shows a sentence that its sources do not support.

## What the system never does

| Never | How it is enforced |
| --- | --- |
| Gives treatment, dosing or management advice | Treatment sentences are removed from the knowledge base at ingestion; treatment requests are flagged and get no knowledge answer; the verifier rejects any sentence with treatment language or a dose |
| States that a patient has a disease | The composer only writes "estimated probability among the shortlisted candidates"; the verifier rejects "has been diagnosed", "the diagnosis is", "confirms" and similar phrases |
| Shows an unsupported sentence | Every sentence needs a citation that exists, at least 40% of its content words in the cited text, and every number, identifier and gene symbol present there |
| Follows instructions found in a note | The pipeline is deterministic: no model reads the note as instructions. Override attempts are flagged and shown to the user |
| Logs identifiers | Redaction runs before the audit log and before any display of the note |
| Analyses during a crisis or an emergency | Crisis disclosures and present tense emergencies stop the analysis and show a fixed message with how to get help |
| Uses family history as the patient's own phenotype | Findings after a relative mention are classed as family history unless a self reference follows |

## The guard

Redaction covers NHS numbers, email addresses, telephone numbers, UK postcodes, dates, hospital numbers and names introduced by a title or a name field. Ages such as "a 6 year old boy" are kept, because age matters to the differential.

| Flag | Trigger | Effect |
| --- | --- | --- |
| treatment | dosing, prescribing, treatment, therapy or management words in the question | no knowledge answer; a fixed message |
| override | an attempt to change the instructions, in the question or in the note | analysis proceeds normally; a fixed message |
| crisis | a disclosure of self harm or suicidal thoughts | analysis blocked; Samaritans 116 123 shown |
| emergency | present tense signs of an emergency ("not breathing", "is in status epilepticus") | analysis blocked; told to follow local emergency procedures |
| certainty | a demand to confirm a diagnosis | analysis proceeds as a differential; a fixed message |
| off topic | no phenotype, disease, gene or clinical term | no answer; a fixed message |

## The verifier

`agent/verifier.py` checks each sentence against the text of its cited sources:

| Check | Fails when |
| --- | --- |
| cited | a non policy sentence cites nothing |
| exists | a cited source is not a passage, the patient record or the model output |
| lexical | fewer than 40% of its content words appear in the cited text (connective words the composer adds are excluded) |
| numbers | a number does not appear in the cited text |
| identifiers | a disease, phenotype or Mondo identifier does not appear in the cited text |
| genes | a gene symbol does not appear in the cited text |
| no_treatment | treatment words or a dose pattern |
| no_certainty | a claim that the patient has a disease or that a diagnosis is confirmed |

Policy sentences come from a fixed list and are exempt from citation. The blind evaluation corrupts verified sentences in eight ways and reports how many are caught. The known gap: the verifier checks words and numbers, not meaning, so a citation to a different passage that shares enough words can pass.

## The audit chain

Every run appends a record to `data/audit/audit_log.jsonl`. The record holds the redacted input, the flags, the recorded terms, the differential with probabilities, the verification count and the strict paper if any. Each record includes the SHA256 of the previous one, so `verify_chain` (and `GET /v1/audit/verify`) finds the first altered or removed record.

## The optional language model writer

Off by default. When enabled, it rewrites the answer to a free text question from the same retrieved passages and must cite them sentence by sentence. Its sentences pass the same verifier. If fewer than half pass, the deterministic answer is used, and the result records which writer produced the answer. The evaluation never uses it.

## Known gaps

* The verifier checks surface support, not meaning.
* Guard rules are regular expressions; unusual phrasing can slip past them. The fixed paths are a safety net, not a substitute for clinical governance.
* Probabilities are calibrated on published cases and are relative to the shortlist. They are not the probability that a real patient has the disease.
* The browser engine has no passage knowledge base and does not run the verifier; its summary says so.
