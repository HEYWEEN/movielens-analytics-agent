import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis_pipeline import _normalize_config, verify_analysis_manifest
from movielens_agent.analysis.classification import HISTORY_START, build_features
from movielens_agent.analysis.common import Dataset, GENRES
from movielens_agent.analysis.profiles import _reduce
from movielens_agent.storage.artifacts import file_sha256, publish_manifest


class AnalysisPipelineTests(unittest.TestCase):
    def test_parameter_validation_and_threshold_selection(self):
        self.assertEqual(_normalize_config({"n_clusters": 4, "target_dimensions": 3}),
                         {"n_clusters": 4, "target_dimensions": 3,
                          "information_threshold": None})
        self.assertEqual(_normalize_config({"information_threshold": 0.5})["target_dimensions"], None)
        with self.assertRaisesRegex(ValueError, "不能同时"):
            _normalize_config({"target_dimensions": 3, "information_threshold": 0.6})
        with self.assertRaisesRegex(ValueError, "2 到 10"):
            _normalize_config({"n_clusters": 1})
        rng = np.random.default_rng(3)
        x = rng.random((50, 5), dtype=np.float32)
        for algorithm in ("pca", "nmf"):
            model, coordinates, seconds, r2, energy, attempted = _reduce(x, algorithm, None, 0.5)
            self.assertGreaterEqual(r2, 0.5)
            self.assertGreaterEqual(coordinates.shape[1], 2)
            self.assertGreaterEqual(energy, r2)
            self.assertEqual(attempted[-1], coordinates.shape[1])

    def test_same_timestamp_and_future_labels_do_not_enter_event_features(self):
        flag = np.zeros(len(GENRES), dtype=np.float32)
        flag[GENRES.index("Drama")] = 1
        data = Dataset(
            source_dir=Path("."), source_manifest={}, t1=2, t2=3,
            users={1: ("F", 25, 3)},
            movies={1: {"title": "A (2000)", "genres": ["Drama"], "flags": flag, "year": 2000},
                    2: {"title": "B (2000)", "genres": ["Drama"], "flags": flag, "year": 2000}},
            user_ids=np.array([1, 1, 1, 1], dtype=np.int32),
            movie_ids=np.array([1, 2, 1, 2], dtype=np.int32),
            ratings=np.array([5, 1, 4, 5], dtype=np.uint8),
            timestamps=np.array([1, 1, 2, 3], dtype=np.int64),
            split={"train": np.array([0, 1, 2]), "validation": np.array([3]),
                   "test": np.array([], dtype=np.int64)},
        )
        features, frozen = build_features(data)
        self.assertEqual(features[0, HISTORY_START], 0)
        self.assertEqual(features[1, HISTORY_START], 0)
        self.assertAlmostEqual(features[2, HISTORY_START], np.log1p(2))
        self.assertAlmostEqual(features[3, HISTORY_START], np.log1p(3))
        self.assertEqual(int(frozen["user_count"][1]), 3)
        changed = Dataset(**{**vars(data), "ratings": np.array([5, 1, 4, 1], dtype=np.uint8)})
        changed_features, _ = build_features(changed)
        np.testing.assert_array_equal(features, changed_features)

    def test_analysis_manifest_detects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            output.mkdir()
            upstream = Path(directory) / "source"
            upstream.mkdir()
            for name in ("users_clean.dat", "movies_clean.dat", "ratings_clean.dat"):
                (upstream / name).write_text("sample", encoding="utf-8")
            upstream_report = {"task_id": "source", "raw_data_version": "raw",
                               "clean_data_version": "clean", "rule_version": "r",
                               "score_config_version": "s", "engine": "local-verification",
                               "time_boundaries": {"T1": 2, "T2": 3}}
            (upstream / "report.json").write_text(json.dumps(upstream_report), encoding="utf-8")
            publish_manifest(upstream, upstream_report)
            names = ("report.json", "models.joblib", "profiles.json", "clusters.json", "coordinates.json")
            source = {"task_id": "source", "clean_data_version": "clean",
                      "source_dir": str(upstream), "source_manifest_sha256": file_sha256(upstream / "manifest.json"),
                      "T1": 2, "T2": 3}
            report = {"status": "completed", "analysis_version": "v1", "source": source,
                      "model_bundle_version": "model"}
            (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
            for name in names[1:]:
                (output / name).write_text("present", encoding="utf-8")
            manifest = {"tool_id": "analysis.iteration2", "analysis_version": "v1",
                        "source": source, "model_bundle_version": "model",
                        "artifacts": [{"name": name, "bytes": (output / name).stat().st_size,
                                       "sha256": file_sha256(output / name)} for name in names]}
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_analysis_manifest(output)["analysis_version"], "v1")
            (output / "coordinates.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coordinates.json"):
                verify_analysis_manifest(output)
            (output / "coordinates.json").write_text("present", encoding="utf-8")
            (upstream / "ratings_clean.dat").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ratings_clean.dat"):
                verify_analysis_manifest(output)


if __name__ == "__main__":
    unittest.main()
