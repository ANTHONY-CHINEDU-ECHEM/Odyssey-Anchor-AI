# Processed tables

Written by `make data` (`odyssey_anchor/ingest/build.py`). Each table is listed with its row count and SHA256 in `MANIFEST.json`, along with the HPO annotation release, the phenopacket store release and the number of treatment sentences removed from Mondo definitions.

| Table | Content |
| --- | --- |
| `hpo_terms.parquet` | every HPO term: label, definition, synonyms, cross references, obsolete flag, organ system |
| `hpo_edges.parquet` | is_a edges between current terms |
| `hpo_alt.parquet` | alternative and obsolete identifiers mapped to current ones |
| `annotations.parquet` | disease annotations with smoothed frequency, NOT flag and reference lists |
| `mondo_classes.parquet`, `mondo_edges.parquet`, `mondo_xrefs.parquet` | Mondo identity, hierarchy and cross references |
| `disease_genes.parquet` | gene to disease links |
| `diseases.parquet` | one row per annotated disease, joined to Mondo, genes, inheritance and onset |
| `cases.parquet` | one row per published patient: observed and excluded terms, diagnosis, gene, paper |

See [docs/data_card.md](../../docs/data_card.md).
