"""Content-verified publication manifest for iteration outputs."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_manifest(output_dir, report):
    output_dir = Path(output_dir)
    names = ("users_clean.dat", "movies_clean.dat", "ratings_clean.dat", "quarantine.jsonl",
             "unresolved_conflicts.jsonl", "before_report.json", "after_report.json", "report.json")
    artifacts = []
    for name in names:
        path = output_dir / name
        if path.is_file():
            artifacts.append({"name": name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
        elif name in {"users_clean.dat", "movies_clean.dat", "ratings_clean.dat", "report.json"} or (
            report.get("engine") == "hadoop-streaming" and "phase_jobs" in report
            and name in {"before_report.json", "after_report.json"}
        ):
            raise RuntimeError(f"Required output missing: {name}")
    manifest = {
        "schema_version": 1,
        "task_id": report["task_id"],
        "tool_id": "governance.clean_and_assess",
        "raw_data_version": report["raw_data_version"],
        "clean_data_version": report["clean_data_version"],
        "rule_version": report["rule_version"],
        "score_config_version": report["score_config_version"],
        "time_boundaries": report["time_boundaries"],
        "engine": report["engine"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify_manifest(output_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = output_dir / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact verification failed: {item['name']}")
    return manifest
