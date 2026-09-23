import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import normalize, process, version_for_files
from runner import REGISTRY, resolve_configuration
from server import agent_chat, requested_configuration

HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    def test_rules_and_references(self):
        records = [
            ("users", "1::M::25::12::00501"),
            ("users", "1::M::25::12::00501"),
            ("users", "6041::F::25::1::12345"),
            ("movies", "1::Film (2000)::Drama"),
            ("ratings", "1::1::4::978307200000"),
            ("ratings", "1::1::4::978307200000"),
            ("ratings", "1::1::2::978307200000"),
            ("ratings", "2::1::5::978307200"),
        ]
        output = []
        report = process(records, lambda kind, payload: output.append((kind, payload)))
        self.assertEqual(report["clean"]["counts"]["users"]["rows"], 1)
        self.assertEqual(report["clean"]["counts"]["ratings"]["rows"], 1)
        self.assertEqual(report["issues"]["ratings.orphan_reference"], 1)
        self.assertEqual(report["issues"]["ratings.conflicting_event"], 1)
        self.assertEqual(report["disposition"]["by_table"]["ratings"]["repaired_retained"], 1)
        self.assertEqual(report["disposition"]["by_table"]["ratings"]["raw_rows"], 4)
        self.assertEqual(report["disposition"]["by_table"]["ratings"]["retained_rows"], 1)
        self.assertEqual(report["score_evidence"]["raw"]["Accurate"]["denominator"], 8)
        self.assertEqual(report["score_evidence"]["clean"]["Unique"],
                         {"numerator": 3, "denominator": 3})
        self.assertIn(("ratings", "1::1::4::978307200"), output)
        self.assertIn(("users", "1::M::25::12::00501"), output)

    def test_bad_time_and_shape(self):
        self.assertEqual(normalize("ratings", "1::1::4::9999999999999")[1], "invalid_domain")
        self.assertIsNone(normalize("ratings", "1::1::4::1046454590")[1])  # Feb 28, 17:49 UTC
        self.assertEqual(normalize("ratings", "1::1::4::1046476800")[1], "invalid_domain")  # Mar 1
        self.assertEqual(normalize("movies", "1::No genre")[1], "field_count")

    def test_normalized_duplicate_is_not_counted_as_retained_repair(self):
        records = [("users", "1::M::25::12::00501"),
                   ("movies", "1::Film (2000)::Drama"),
                   ("ratings", "1::1::4::978307200"),
                   ("ratings", "1::1::4::978307200000")]
        report = process(records, lambda *_: None)
        self.assertEqual(report["disposition"]["repaired_retained"], 0)
        self.assertEqual(report["disposition"]["exact_duplicates"], 1)

    def test_equal_frequency_conflict_remains_unverified_after_rescoring(self):
        records = [("users", "1::M::25::12::00501"),
                   ("users", "1::F::56::12::00501"),
                   ("movies", "1::Film (2000)::Drama"),
                   ("ratings", "1::1::4::978307200")]
        output = []
        report = process(records, lambda kind, payload: output.append((kind, payload)))
        self.assertEqual(report["unresolved_ties"]["by_table"]["users"], 1)
        self.assertEqual(report["raw"]["counts"]["users"]["ambiguous"], 2)
        self.assertEqual(report["clean"]["counts"]["users"]["ambiguous"], 1)
        self.assertLess(report["clean"]["scores"]["Accurate"], 100)
        self.assertEqual(report["disposition"]["by_table"]["users"]["retained_unverified_ties"], 1)
        marker = next(payload for kind, payload in output if kind == "uncertain")
        cleaned = [(kind, payload) for kind, payload in output if kind in {"users", "movies", "ratings"}]
        rescored = process(cleaned + [("uncertain", marker)], lambda *_: None)
        self.assertEqual(rescored["raw"]["scores"], report["clean"]["scores"])

    def test_streaming_entry_points(self):
        mapped = []
        for table, value in (("users", "1::M::25::12::00501"),
                             ("movies", "1::Café (2000)::Drama"),
                             ("ratings", "1::1::5::978307200")):
            env = dict(os.environ, map_input_file=f"/input/{table}.dat")
            result = subprocess.run([sys.executable, str(HERE / "streaming.py"), "mapper"],
                                    input=(value + "\n").encode("latin-1"), env=env,
                                    capture_output=True, check=True)
            mapped.append(result.stdout)
        result = subprocess.run([sys.executable, str(HERE / "streaming.py"), "reducer"],
                                input=b"".join(mapped), capture_output=True, check=True)
        output = result.stdout.decode().splitlines()
        self.assertTrue(any(line == "movies\t1::Café (2000)::Drama" for line in output))
        report = json.loads(next(line.split("\t", 1)[1] for line in output if line.startswith("report\t")))
        self.assertEqual(report["clean"]["counts"]["ratings"]["rows"], 1)
        before = subprocess.run([sys.executable, str(HERE / "streaming.py"), "reducer", "before"],
                                input=b"".join(mapped), capture_output=True, check=True)
        self.assertEqual(len(before.stdout.decode().splitlines()), 1)
        self.assertTrue(before.stdout.startswith(b"report\t"))

    def test_registered_configuration_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            paths = {table: data_dir / f"{table}.dat" for table in ("users", "movies", "ratings")}
            for path in paths.values():
                path.write_text("", encoding="latin-1")
            synthetic_version = version_for_files(paths)
            with patch.dict(REGISTRY["data_versions"],
                            {synthetic_version: {"path": str(data_dir)}}), \
                 patch.dict(REGISTRY["defaults"], {"data_version": synthetic_version}):
                _, config = resolve_configuration()
                self.assertEqual(config["rule_version"], "ml1m-rules-v4")
                with self.assertRaisesRegex(ValueError, "未登记的数据版本"):
                    resolve_configuration(data_version="unknown")
                with self.assertRaisesRegex(ValueError, "未登记或不可执行的评分配置"):
                    resolve_configuration(score_config_version="unknown")
                self.assertEqual(requested_configuration("请用规则版本 ml1m-rules-v4 清洗", {})["rule_version"],
                                 "ml1m-rules-v4")
                self.assertEqual(agent_chat("请用数据版本 unknown 清洗")["error"], "unsupported_configuration")
                self.assertEqual(agent_chat("你好", task_id="unknown")["error"], "unsupported_intent")


if __name__ == "__main__":
    unittest.main()
