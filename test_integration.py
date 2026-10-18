"""End to end tests on the real knowledge base. Skipped until make data index train has run."""
import json

import pandas as pd
import pytest

from odyssey_anchor import config
from odyssey_anchor.agent.copilot import Request
from tests.conftest import requires_data

MARFAN = ("Tall 14 year old boy with arachnodactyly, ectopia lentis, pectus excavatum and aortic root aneurysm. "
          "Joint hypermobility and striae. No intellectual disability. His father has myopia.")


@requires_data
def test_marfan_note_end_to_end(copilot):
    r = copilot.run(Request(note=MARFAN, sex="MALE", age_years=14, question="How is the leading candidate inherited?"))
    names = [d["name"] for d in r["differential"][:5]]
    assert any("Marfan" in n for n in names), names
    assert "HP:0000545" in {t["id"] for t in r["patient"]["family"]}
    assert "HP:0001249" in {t["id"] for t in r["patient"]["excluded"]}
    assert r["verification"]["passed"] == r["verification"]["total"] or r["verification"]["dropped"]
    for s in r["sentences"]:
        assert s["cites"] or s["role"] == "policy"
        for c in s["cites"]:
            assert c in r["sources"]
    assert any(s["role"] == "answer" and "autosomal dominant" in s["text"].lower() for s in r["sentences"])
    probs = [d["probability"] for d in r["differential"]]
    assert probs == sorted(probs, reverse=True)


@requires_data
def test_strict_mode_removes_source_paper(copilot):
    split = pd.read_parquet(config.EVAL / "case_split.parquet")
    c = split[split.split.eq("dev") & split.pmid.notna()].iloc[0]
    eng = copilot.engine
    pat = eng.patient(list(c.observed), list(c.excluded), c.sex)
    released = eng.score(pat)
    strict = eng.score(pat, pmid=c.pmid, strict=True)
    if strict.strict_diseases:
        g = eng.row[strict.strict_diseases[0]]
        assert strict["lr"][g] != released["lr"][g] or strict["bma"][g] != released["bma"][g]


@requires_data
def test_frozen_hash_holds():
    from odyssey_anchor.eval import freeze
    frozen = freeze.verify()
    assert frozen["blind_cases"] > 1000
    runs = json.loads((config.EVAL / "blind_runs.json").read_text())["runs"]
    assert len(runs) <= 1, "the blind set must be scored once"


@requires_data
def test_api_endpoints(copilot, monkeypatch):
    from fastapi.testclient import TestClient
    from odyssey_anchor.api import app as api
    monkeypatch.setattr(api, "copilot", lambda: copilot)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/v1/analyse", json={"note": MARFAN, "sex": "MALE", "age_years": 14})
    assert r.status_code == 200 and r.json()["differential"]
    assert client.post("/v1/analyse", json={}).status_code == 400
    a = client.post("/v1/ask", json={"question": "What conditions are caused by variants in FBN1?"}).json()
    assert any("FBN1" in s["text"] for s in a["sentences"])
    e = client.post("/v1/extract", json={"note": "No seizures but hypotonia."}).json()
    assert e["terms"]["observed"] == ["HP:0001252"] and e["terms"]["excluded"] == ["HP:0001250"]
    assert client.get("/v1/terms/search", params={"q": "ectopia"}).json()[0]["id"] == "HP:0001083"
    assert client.get("/v1/diseases/OMIM:154700").json()["genes"] == ["FBN1"]
    assert client.get("/v1/diseases/NOPE:1").status_code == 404
    assert client.get("/v1/audit/verify").json()["valid"]


@requires_data
def test_knowledge_base_counts():
    stats = json.loads((config.INDEX / "passages_stats.json").read_text())
    assert stats["total"] > 100_000
    passages = pd.read_parquet(config.INDEX / "passages.parquet", columns=["pid", "text"])
    assert passages.pid.is_unique
    # No treatment sentence reaches the knowledge base from Mondo definitions.
    over = passages[passages.pid.str.endswith("|overview")]
    assert not over.text.str.contains(r"\bTreatment is\b", regex=True).any()
