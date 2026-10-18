# Deployment

## Requirements

Python 3.11 or later, Node 20 or later (only for the browser engine tests), about 2 GB of free disk and 4 GB of memory. The pipeline runs on two CPU cores; nothing needs a GPU.

## From a clean clone

```
pip install ".[dev]"
make data           # download five public files (about 200 MB) and build the tables
make freeze         # freeze and hash the evaluation sets
make index          # knowledge base, text indexes, passage ranker
make tune features train
make evaluate       # development evaluation
make teaching figures bundle web
make serve          # http://localhost:8000
```

`make all` runs the same sequence. The blind set is scored only by `make evaluate_blind`, which checks the frozen hash and appends a timestamp to `data/eval/blind_runs.json`, so any repeat run is visible.

Approximate times on two cores: data 1 minute, index 7 minutes, tune 3 minutes, features 15 minutes, train 12 minutes, development evaluation 40 minutes, blind evaluation 60 minutes.

## Docker

```
docker compose up --build
```

The image installs the package and serves the API and interface on port 8000. If `data/processed` is missing it runs `make data index` while building. The audit log lives on a named volume.

## The optional language model writer

```
export ODYSSEY_LLM=anthropic
export ANTHROPIC_API_KEY=...
export ODYSSEY_LLM_MODEL=claude-sonnet-5-5
make serve
```

Only the free text answer to a question is affected. The writer sees the retrieved passages and nothing else, must cite every sentence, and its output passes the verifier (see [safety.md](safety.md)).

## Static hosting

`make bundle web` writes `web/dist/index.html`, one self contained page with the browser engine, the knowledge bundle, the teaching cases and the evaluation report. It can be served from any static host or opened from disk; it makes no network calls apart from fonts. `web/dist_single/odyssey_anchor.html` is the same page body without the document wrapper, for embedding.

## Updating the data

A new HPO or Mondo release changes the knowledge base, not the protocol. Rerun `make data index tune features train evaluate`. The frozen blind set refers to patients and diseases by identifier, so `make freeze` should not be rerun unless a new evaluation is intended. If a new release adds patients, freeze a new blind set and record why.
