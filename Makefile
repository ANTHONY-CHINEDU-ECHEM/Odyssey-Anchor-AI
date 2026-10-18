PY ?= python3
RUN = $(PY) -m odyssey_anchor

.PHONY: all data freeze index tune features train evaluate evaluate_fast evaluate_blind teaching figures bundle web test test_py test_js serve docker clean_models

## Full pipeline from a clean checkout (the blind set is scored separately, once)
all: data freeze index tune features train evaluate teaching figures bundle web

data:            ## download the public sources and build the processed tables
	$(RUN) data

freeze:          ## freeze and hash the development and blind sets (run before any index or model)
	$(RUN) freeze

index:           ## knowledge base passages, disease text indexes, passage ranker
	$(RUN) index

tune:            ## likelihood ratio settings on development cases
	$(RUN) tune

features:        ## shortlist features for every development case, strict protocol
	$(RUN) features

train:           ## fused ranker with grouped cross validation and temperature calibration
	$(RUN) train

evaluate:        ## development evaluation
	$(RUN) evaluate

evaluate_fast:
	$(RUN) evaluate --fast

evaluate_blind:  ## score the frozen blind set; every run is appended to data/eval/blind_runs.json
	$(RUN) evaluate_blind

teaching:        ## export teaching cases from the blind set for the interface
	$(RUN) teaching

figures:         ## draw every figure in reports/figures
	$(RUN) figures

bundle:          ## browser bundle and parity fixture
	$(RUN) bundle

web:             ## build the single page interface into web/dist
	$(PY) web/build.py

test: test_py test_js

test_py:
	$(PY) -m pytest -q tests

test_js:
	node --test web/tests/

serve:           ## API and interface on http://localhost:8000
	$(RUN) serve

docker:
	docker compose up --build

clean_models:
	rm -f data/models/*.parquet data/models/*.json data/models/*.txt
