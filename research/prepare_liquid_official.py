#!/usr/bin/env python3
"""Download and structurally audit the official LIQUID datasets.

This script deliberately reports schema/count/hash information only. It does not
print question, context, or answer text, which keeps the future evaluation split
uninspected while we establish provenance and compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


URLS = {
    "wiki": "http://nlp.dmis.korea.edu/projects/liquid-lee-et-al-2023/liquid-wiki-140k.json",
    "pubmed": "http://nlp.dmis.korea.edu/projects/liquid-lee-et-al-2023/liquid-pubmed-140k.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    temp = path.with_suffix(path.suffix + ".part")
    request = Request(url, headers={"User-Agent": "EQUAL-RAG-research/1.0"})
    with urlopen(request, timeout=300) as response, temp.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
    temp.replace(path)


def shape(value):
    if isinstance(value, dict):
        return {key: type(item).__name__ for key, item in value.items()}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "item_type": type(value[0]).__name__ if value else None}
    return {"type": type(value).__name__}


def audit(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    report = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "top_level": shape(value),
    }
    if isinstance(value, list) and value:
        report["first_item"] = shape(value[0])
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, list):
                report[f"{key}_length"] = len(item)
                if item:
                    report[f"{key}_first_item"] = shape(item[0])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for domain, url in URLS.items():
        path = output_dir / f"liquid-{domain}-140k.json"
        download(url, path)
        reports[domain] = audit(path)
    report_path = output_dir / "source_audit.json"
    report_path.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
