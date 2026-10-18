"""Append only, hash chained audit log of every copilot run.

Each record stores the redacted input, the flags, the differential, the
verification outcome and the hash of the previous record. Changing or removing
any record breaks every hash after it, which `verify_chain` detects.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import config

GENESIS = "0" * 64
_LOCK = threading.Lock()


def _digest(prev: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path | None = None):
        self.path = path or (config.AUDIT / "audit_log.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode().strip().splitlines()
        return json.loads(tail[-1])["hash"]

    def append(self, payload: dict) -> str:
        with _LOCK:
            prev = self.last_hash()
            payload = dict(payload)
            payload["logged_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            h = _digest(prev, payload)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prev": prev, "hash": h, "payload": payload}, default=str) + "\n")
            return h

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify_chain(self) -> dict:
        prev = GENESIS
        for i, rec in enumerate(self.records()):
            if rec["prev"] != prev or _digest(prev, rec["payload"]) != rec["hash"]:
                return {"valid": False, "broken_at": i}
            prev = rec["hash"]
        return {"valid": True, "records": len(self.records()), "head": prev}
