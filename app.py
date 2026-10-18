"""FastAPI service for the copilot and the knowledge base.

Endpoints
  GET  /health                     liveness and data versions
  POST /v1/analyse                 note and/or HPO terms (+ optional question) to a cited differential
  POST /v1/ask                     a knowledge question, optionally with the current differential as context
  POST /v1/extract                 phenotype extraction only, with spans and cues
  GET  /v1/terms/search?q=         phenotype term lookup for the interface
  GET  /v1/diseases/{disease_id}   disease overview with genes, inheritance and organ system passages
  GET  /v1/passages/{pid}          one knowledge base passage
  GET  /v1/cases                   teaching cases from the blind set (strict protocol)
  GET  /v1/reports/{name}          evaluation reports
  GET  /v1/audit/verify            integrity of the audit chain
  GET  /                           the web interface
"""
from __future__ import annotations

import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, config
from ..agent.copilot import Copilot, Request
from .schemas import AnalyseIn, AskIn, ExtractIn

app = FastAPI(title="Odyssey Anchor AI", version=__version__,
              description="Phenotype driven rare disease differential diagnosis with a verified graph retrieval copilot.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@lru_cache(maxsize=1)
def copilot() -> Copilot:
    return Copilot()


def _clean(res: dict) -> dict:
    return json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


@app.get("/health")
def health() -> dict:
    manifest = json.loads((config.PROCESSED / "MANIFEST.json").read_text())
    return {"status": "ok", "version": __version__, "hpoa_version": manifest.get("hpoa_version"),
            "phenopacket_release": manifest.get("phenopacket_release"),
            "diseases": manifest["tables"]["diseases"]["rows"]}


@app.post("/v1/analyse")
def analyse(body: AnalyseIn) -> dict:
    if not body.note and not body.observed and not body.question:
        raise HTTPException(400, "Provide a note, observed terms or a question")
    res = copilot().run(Request(note=body.note or "", observed=body.observed, excluded=body.excluded,
                                question=body.question or "", sex=body.sex, age_years=body.age_years,
                                strict_pmid=body.strict_pmid))
    return _clean(res)


@app.post("/v1/ask")
def ask(body: AskIn) -> dict:
    res = copilot().run(Request(question=body.question, context=body.context))
    return _clean({k: res[k] for k in ("run_id", "guard", "answer", "sentences", "verification", "sources", "timings_ms")})


@app.post("/v1/extract")
def extract(body: ExtractIn) -> dict:
    cop = copilot()
    findings = cop.recognizer.extract(body.note)
    return {"findings": cop.recognizer.as_dicts(findings), "terms": cop.recognizer.to_terms(findings)}


@app.get("/v1/terms/search")
def term_search(q: str = Query(..., min_length=2), limit: int = 12) -> list[dict]:
    cop = copilot()
    onto = cop.engine.onto
    ql = q.lower()
    hits = []
    for i, (tid, label) in enumerate(zip(onto.ids, onto.labels)):
        if not onto.phenotype_mask[i]:
            continue
        lab = label.lower()
        if ql in lab or ql == tid.lower():
            hits.append((0 if lab.startswith(ql) else 1, 0, len(lab), tid, label))
        elif any(ql in s.lower() for s in onto.synonyms[i]):
            hits.append((2, 0, len(lab), tid, label))
    hits.sort()
    return [{"id": h[3], "label": h[4]} for h in hits[:limit]]


@app.get("/v1/diseases/{disease_id}")
def disease(disease_id: str) -> dict:
    cop = copilot()
    p = cop.passages.p
    rows = p[p.disease_id.eq(disease_id)]
    if rows.empty:
        raise HTTPException(404, "Unknown disease")
    return {"disease_id": disease_id, "name": cop.names.get(disease_id),
            "genes": list(cop.dgenes.get(disease_id, [])), "inheritance": list(cop.inheritance.get(disease_id, [])),
            "passages": rows[["pid", "title", "text"]].to_dict("records")}


@app.get("/v1/passages/{pid:path}")
def passage(pid: str) -> dict:
    cop = copilot()
    if pid not in cop.ptext:
        raise HTTPException(404, "Unknown passage")
    return {"pid": pid, "text": cop.ptext[pid]}


@app.get("/v1/cases")
def cases() -> JSONResponse:
    path = config.REPORTS / "teaching_cases.json"
    if not path.exists():
        raise HTTPException(404, "Teaching cases have not been exported")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/v1/reports/{name}")
def report(name: str) -> JSONResponse:
    allowed = {"evaluation_blind", "evaluation_dev", "lr_tuning"}
    if name not in allowed:
        raise HTTPException(404, "Unknown report")
    path = config.REPORTS / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, "Report not generated yet")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/v1/audit/verify")
def audit_verify() -> dict:
    return copilot().audit.verify_chain()


_dist = config.WEB / "dist"
if _dist.exists():
    app.mount("/static", StaticFiles(directory=_dist), name="static")


@app.get("/")
def index():
    page = _dist / "index.html"
    if not page.exists():
        return {"message": "Build the interface with make web"}
    return FileResponse(page)
