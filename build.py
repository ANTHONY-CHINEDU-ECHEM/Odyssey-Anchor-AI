"""Build the single page interface.

Inlines the stylesheet, the browser engine, the recogniser, the application,
the teaching cases, the evaluation report, the illustrative notes and the
gzip compressed knowledge bundle into one self contained page:

  web/dist/index.html                 full document, served by the API at /
  web/dist_single/odyssey_anchor.html page body for publishing as an artifact
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SRC = WEB / "src"


def module_to_script(code: str) -> str:
    code = re.sub(r"^import .*?;\s*$", "", code, flags=re.M)
    return re.sub(r"^export (class|function|const)", r"\1", code, flags=re.M)


def safe_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def report() -> dict:
    rep = {}
    blind = ROOT / "reports" / "evaluation_blind.json"
    if blind.exists():
        b = json.loads(blind.read_text())
        b.pop("retrieval_misses", None)
        rep["blind"] = b
    for name, path in [("frozen", ROOT / "data" / "eval" / "FROZEN.json"),
                       ("manifest", ROOT / "data" / "processed" / "MANIFEST.json"),
                       ("kb", ROOT / "data" / "index" / "passages_stats.json"),
                       ("leak_stats", ROOT / "reports" / "leakage_stats.json")]:
        if path.exists():
            rep[name] = json.loads(path.read_text())
    return rep


def build() -> dict:
    page = (SRC / "index.html").read_text()
    css = (SRC / "styles.css").read_text()
    engine = module_to_script((SRC / "engine.js").read_text())
    recog = module_to_script((SRC / "recognizer.js").read_text())
    app = (SRC / "app.js").read_text()
    script = ("(function(){\n" + engine + "\n" + recog +
              "\nwindow.OA = { Engine, Recognizer, redact };\n})();\n" + app)
    teaching_path = ROOT / "reports" / "teaching_cases.json"
    teaching = json.loads(teaching_path.read_text()) if teaching_path.exists() else []
    vignettes = json.loads((SRC / "vignettes.json").read_text())
    bundle_b64 = base64.b64encode((WEB / "data" / "bundle.json.gz").read_bytes()).decode()
    body = (page.replace("/*STYLES*/", css)
                .replace("/*TEACHING*/", safe_json(teaching))
                .replace("/*REPORT*/", safe_json(report()))
                .replace("/*VIGNETTES*/", safe_json(vignettes))
                .replace("/*BUNDLE*/", bundle_b64)
                .replace("/*SCRIPT*/", script.replace("</script", "<\\/script")))
    (WEB / "dist_single").mkdir(exist_ok=True)
    (WEB / "dist").mkdir(exist_ok=True)
    (WEB / "dist_single" / "odyssey_anchor.html").write_text(body)
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "<style>[hidden]{display:none!important}</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    (WEB / "dist" / "index.html").write_text(full)
    return {"bytes": len(full), "teaching_cases": len(teaching)}


if __name__ == "__main__":
    print(build())
