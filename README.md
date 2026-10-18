# Raw public sources

`make data` downloads five files here from GitHub release assets and records their URL, size, SHA256 and release version in `MANIFEST.json`. The files themselves are not committed.

| File | Source | Size |
| --- | --- | --- |
| `hp.json` | Human Phenotype Ontology, obographs JSON | 22 MB |
| `phenotype.hpoa` | HPO disease annotations | 36 MB |
| `genes_to_phenotype.txt` | HPO gene links | 21 MB |
| `mondo.json` | Mondo Disease Ontology | 108 MB |
| `all_phenopackets.zip` | GA4GH phenopacket store | 19 MB |

To reproduce the published results exactly, compare the SHA256 values in `MANIFEST.json` with the files you download. Newer releases will give slightly different numbers.
