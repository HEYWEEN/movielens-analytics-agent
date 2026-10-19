"""Independent checks for a published iteration-two analysis directory."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from analysis_pipeline import (ANALYSIS_TOOLS, execute_analysis_tool,
                               load_report, verify_analysis_manifest)


def verify(path):
    path = Path(path)
    manifest = verify_analysis_manifest(path)
    report = load_report(path)
    assert sum(report["split"][name]["rows"] for name in ("train", "validation", "test")) > 0
    assert set(report["classification"]["algorithms"]) == {"logistic_regression", "decision_tree"}
    assert set(report["clustering"]["algorithms"]) == {"kmeans", "agglomerative"}
    assert set(report["reduction"]["algorithms"]) == {"pca", "nmf"}
    profile_data = json.loads((path / "profiles.json").read_text(encoding="utf-8"))
    profiles = profile_data["users"]
    coordinates = json.loads((path / "coordinates.json").read_text(encoding="utf-8"))["users"]
    assert len(profiles) == len(coordinates) == report["clustering"]["eligible_users"]
    if int(report["analysis_config_version"].rsplit("v", 1)[1]) >= 4:
        dimension = len(profile_data["feature_names"])
        assert dimension == report["reduction"]["original_dimensions"]
        assert len(profiles[0]["feature_vector"]) == len(profiles[0]["raw_feature_vector"]) == dimension
        assert len(profile_data["scaler"]["scale"]) == dimension
        for algorithm in ("pca", "nmf"):
            assert len(coordinates[0][algorithm]) == report["reduction"]["algorithms"][algorithm]["actual_dimensions"]
    representative_user = profiles[0]["user_id"]
    state = joblib.load(path / "models.joblib")
    training_pairs = state["known_training_pairs"]
    unseen_movie = None
    for movie_id in state["movies"]:
        pair = np.uint64((representative_user << 32) | movie_id)
        position = int(np.searchsorted(training_pairs, pair))
        if position == len(training_pairs) or training_pairs[position] != pair:
            unseen_movie = movie_id
            break
    assert unseen_movie is not None, "No safe unseen user/movie pair for classification smoke check"
    for tool_id in ANALYSIS_TOOLS:
        result = execute_analysis_tool(tool_id, path, user_id=representative_user,
                                       movie_id=unseen_movie, limit=3)
        assert result["status"] == "completed", (tool_id, result)
        assert len(result["results"]) == 1, tool_id
    return {"status": "verified", "analysis_version": manifest["analysis_version"],
            "source_clean_data_version": manifest["source"]["clean_data_version"],
            "split_rows": {name: report["split"][name]["rows"]
                           for name in ("train", "validation", "test")},
            "eligible_users": len(profiles), "artifacts": len(manifest["artifacts"]),
            "verified_tools": len(ANALYSIS_TOOLS), "registered_tools": len(ANALYSIS_TOOLS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), ensure_ascii=False, indent=2))
