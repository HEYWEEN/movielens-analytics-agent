import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from movielens_agent.storage.artifacts import publish_manifest, verify_manifest
from movielens_agent.storage.task_store import TaskStore
from pipeline import version_for_files
from runner import REGISTRY, run_local


class ArchitectureTests(unittest.TestCase):
    def test_local_run_publishes_verifiable_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for table, content in {
                "users": "1::M::25::12::00501\n",
                "movies": "1::Film (2000)::Drama\n",
                "ratings": "1::1::4::978307200\n",
            }.items():
                (data / f"{table}.dat").write_text(content, encoding="latin-1")
            version = version_for_files({table: data / f"{table}.dat" for table in ("users", "movies", "ratings")})
            with patch.dict(REGISTRY["data_versions"], {version: {"path": str(data)}}):
                report = run_local(data_dir=data, runs_dir=root / "runs", task_id="c" * 12,
                                   data_version=version)
            manifest = verify_manifest(root / "runs" / ("c" * 12))
            self.assertEqual(manifest["clean_data_version"], report["clean_data_version"])
            self.assertEqual(len(report["clean_data_version"]), 16)

    def test_task_status_survives_restart_and_interrupted_run_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite3"
            first = TaskStore(path)
            first.put({"task_id": "a" * 12, "status": "queued", "stage": "queued", "events": []})
            first.update("a" * 12, status="running", stage="clean", events=[{"message": "started"}])
            restarted = TaskStore(path)
            task = restarted.get("a" * 12)
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["failed_stage"], "clean")
            self.assertEqual(task["events"], [{"message": "started"}])
            self.assertEqual(restarted.latest_id(), "a" * 12)

    def test_manifest_detects_changed_clean_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for table in ("users", "movies", "ratings"):
                (output / f"{table}_clean.dat").write_text(table, encoding="utf-8")
            report = {"task_id": "b" * 12, "raw_data_version": "raw", "clean_data_version": "clean",
                      "rule_version": "rules", "score_config_version": "scores", "time_boundaries": {"T1": 1, "T2": 2},
                      "engine": "local-verification"}
            (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
            publish_manifest(output, report)
            self.assertEqual(verify_manifest(output)["task_id"], "b" * 12)
            (output / "users_clean.dat").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "users_clean.dat"):
                verify_manifest(output)


if __name__ == "__main__":
    unittest.main()
