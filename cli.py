"""Command line entry point: python -m odyssey_anchor <command>.

Commands follow the order of the pipeline. Each writes its outputs under data/
or reports/ and can be rerun on its own.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config


def cmd_data(_):
    from .ingest import build, download
    download.download_all()
    print(json.dumps(build.run()["tables"], indent=2))


def cmd_freeze(_):
    from .eval import freeze
    print(json.dumps(freeze.run(), indent=2))


def cmd_index(_):
    from .extract.recognizer import Recognizer
    from .knowledge import build
    from .ontology.graph import Ontology
    from .rank.engine import Engine
    from .retrieval import passages
    print(json.dumps(build.run(), indent=2))
    Engine()  # builds and caches the disease text indexes
    onto = Ontology.load()
    _, info = passages.build(onto, Recognizer(onto).lex)
    print(json.dumps(info, indent=2))


def cmd_tune(_):
    from .rank import tune
    print(json.dumps(tune.run()["chosen"], indent=2))


def cmd_features(_):
    from .rank import dataset
    F, R = dataset.run("dev")
    F.to_parquet(config.MODELS / "features_dev.parquet", index=False)
    R.to_parquet(config.MODELS / "base_ranks_dev.parquet", index=False)


def cmd_train(_):
    from .rank import fusion
    meta = fusion.run()
    print(json.dumps({k: meta[k] for k in ("product", "temperature", "cv")}, indent=2))


def cmd_evaluate(args):
    from .eval import evaluate, leakage
    print(json.dumps(leakage.run(), indent=2))
    evaluate.evaluate_dev(fast=args.fast)


def cmd_evaluate_blind(_):
    from .eval import evaluate
    rep = evaluate.evaluate_blind()
    print(json.dumps(rep["headline"]["concept"], indent=2))


def cmd_teaching(_):
    from .export import teaching
    print(teaching.run())


def cmd_figures(_):
    from .eval import figures
    figures.run()


def cmd_bundle(_):
    from .export import bundle
    print(json.dumps(bundle.run(), indent=2))


def cmd_serve(args):
    import uvicorn
    uvicorn.run("odyssey_anchor.api.app:app", host=args.host, port=args.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="odyssey_anchor", description="Odyssey Anchor AI pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in [("data", cmd_data), ("freeze", cmd_freeze), ("index", cmd_index), ("tune", cmd_tune),
                     ("features", cmd_features), ("train", cmd_train), ("evaluate_blind", cmd_evaluate_blind),
                     ("teaching", cmd_teaching), ("figures", cmd_figures), ("bundle", cmd_bundle)]:
        sub.add_parser(name).set_defaults(fn=fn)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--fast", action="store_true")
    ev.set_defaults(fn=cmd_evaluate)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    config.ensure_dirs()
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
