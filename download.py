"""Download the public sources and record their fingerprints.

The manifest stores, for every file, the URL, the size, the SHA256 and the
release version found inside the file, so any later run can show whether it
used the same data.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "odyssey-anchor/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def release_version(name: str, path: Path) -> str | None:
    """Read the release version stamped inside each source file."""
    try:
        if name == "phenotype.hpoa":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#version") or line.startswith("#date"):
                        return line.split(":", 1)[1].strip()
                    if not line.startswith("#"):
                        break
        if name in ("hp.json", "mondo.json"):
            with open(path, encoding="utf-8") as f:
                head = f.read(4000)
            key = '"version" : "'
            if key in head:
                return head.split(key, 1)[1].split('"', 1)[0]
    except OSError:
        return None
    return None


def download_all(force: bool = False) -> dict:
    config.ensure_dirs()
    manifest = {"downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    for name, url in config.SOURCES.items():
        dest = config.RAW / name
        if force or not dest.exists():
            print(f"downloading {name}")
            fetch(url, dest)
        manifest["files"][name] = {
            "url": url,
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
            "release": release_version(name, dest),
        }
    out = config.RAW / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    print(json.dumps(download_all(), indent=2))
