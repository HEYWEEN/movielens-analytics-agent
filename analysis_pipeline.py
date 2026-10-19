"""Iteration-two analysis runner and six independently callable Agent tools.

All model training and user profiles use the verified iteration-one clean output.
No query fabricates a result when the analysis artifact is absent or invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from movielens_agent.analysis import ANALYSIS_VERSION
from movielens_agent.analysis.classification import (FEATURE_NAMES, build_features, pair_features,
                                                     train_classifiers)
from movielens_agent.analysis.common import load_dataset, split_summary
from movielens_agent.analysis.profiles import fit_profiles
from movielens_agent.storage.artifacts import file_sha256, verify_manifest

BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE / "runs" / "e95ea259cb07"
DEFAULT_OUTPUT = BASE / "runs" / "iteration2" / "default"
ANALYSIS_TOOLS = {
    "classification.logistic_regression": ("classification", "logistic_regression"),
    "classification.decision_tree": ("classification", "decision_tree"),
    "clustering.kmeans": ("clustering", "kmeans"),
    "clustering.agglomerative": ("clustering", "agglomerative"),
    "reduction.pca": ("reduction", "pca"),
    "reduction.nmf": ("reduction", "nmf"),
}
ARTIFACTS = ("report.json", "models.joblib", "profiles.json", "clusters.json", "coordinates.json")


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _source_record(data):
    manifest = data.source_manifest
    return {
        "source_dir": str(data.source_dir),
        "task_id": manifest["task_id"],
        "raw_data_version": manifest["raw_data_version"],
        "clean_data_version": manifest["clean_data_version"],
        "rule_version": manifest["rule_version"],
        "score_config_version": manifest["score_config_version"],
        "engine": manifest["engine"],
        "T1": data.t1, "T2": data.t2,
        "T1_utc": manifest["time_boundaries"]["T1_utc"],
        "T2_utc": manifest["time_boundaries"]["T2_utc"],
        "source_manifest_sha256": file_sha256(data.source_dir / "manifest.json"),
    }


def _normalize_config(config):
    config = dict(config or {})
    allowed = {"n_clusters", "target_dimensions", "information_threshold"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"不支持的分析配置：{', '.join(sorted(unknown))}")
    k = config.get("n_clusters", 5)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 10:
        raise ValueError("n_clusters 必须是 2 到 10 的整数。")
    threshold = config.get("information_threshold")
    if threshold is not None:
        if "target_dimensions" in config:
            raise ValueError("information_threshold 与 target_dimensions 不能同时指定。")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.5 <= threshold < 1:
            raise ValueError("information_threshold 必须介于 0.5（含）和 1（不含）之间。")
        return {"n_clusters": k, "target_dimensions": None,
                "information_threshold": float(threshold)}
    dimensions = config.get("target_dimensions", 2)
    if (isinstance(dimensions, bool) or not isinstance(dimensions, int)
            or not 2 <= dimensions <= 10):
        raise ValueError("target_dimensions 必须是 2 到 10 的整数。")
    return {"n_clusters": k, "target_dimensions": dimensions,
            "information_threshold": None}


def run_analysis(source_dir=DEFAULT_SOURCE, output_dir=DEFAULT_OUTPUT,
                 on_progress=None, config=None) -> dict:
    """Train and publish a complete reproducible analysis job.

    `on_progress(stage, message)` is called at major stages. Output is published
    only after all six algorithms and artifact hashes have completed.
    """
    if on_progress:
        on_progress("validate", "正在核验迭代一产物摘要、版本与时间边界")
    data = load_dataset(source_dir)
    config = _normalize_config(config)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"分析产物目录已包含文件：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split = split_summary(data)
    if on_progress:
        on_progress("features", "正在按事件时间构造分类特征，并冻结 T1 历史状态")
    features, frozen = build_features(data)
    classification, classifiers, classification_baseline = train_classifiers(data, features, on_progress)
    del features
    profile_summary, clustering, reduction, profiles, coordinates, profile_models, train_counts = fit_profiles(
        data, on_progress, config["n_clusters"], config["target_dimensions"],
        config["information_threshold"])
    source = _source_record(data)
    algorithm_config = {
        "classification": "event-prior-v1; LogisticRegression C in {0.3,1.0}; Tree depth in {8,12}",
        "clustering": f"train-profile-37d; k={config['n_clusters']}; min_history=20",
        "reduction": f"PCA/NMF; target={config['target_dimensions']}; threshold={config['information_threshold']}",
        "random_state": 42,
    }
    version_material = json.dumps({"code": ANALYSIS_VERSION, "source": source,
                                   "config": config, "algorithm_config": algorithm_config,
                                   "sklearn": sklearn.__version__}, sort_keys=True)
    analysis_version = hashlib.sha256(version_material.encode()).hexdigest()[:16]
    if on_progress:
        on_progress("publish", "正在保存模型、画像、坐标与评价报告")
    train = data.split["train"]
    known_pairs = np.unique((data.user_ids[train].astype(np.uint64) << 32)
                            | data.movie_ids[train].astype(np.uint64))
    state = {
        "classifiers": classifiers,
        "frozen": frozen,
        "known_training_pairs": known_pairs,
        "users": set(data.users),
        "movies": {mid: {"title": info["title"], "genres": info["genres"]}
                   for mid, info in data.movies.items()},
        "profile_models": profile_models,
    }
    joblib.dump(state, output_dir / "models.joblib", compress=3)
    model_digest = file_sha256(output_dir / "models.joblib")[:16]
    for name, result in classification.items():
        result["model_version"] = f"{model_digest}:{name}"
        result["tool_id"] = f"classification.{name}"
    for name, result in clustering.items():
        result["model_version"] = f"{model_digest}:{name}"
        result["tool_id"] = f"clustering.{name}"
    for name, result in reduction.items():
        result["model_version"] = f"{model_digest}:{name}"
        result["tool_id"] = f"reduction.{name}"
    report = {
        "status": "completed", "schema_version": 1,
        "analysis_version": analysis_version,
        "analysis_config_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "configuration": config,
        "split": {"method": "train: timestamp <= T1; validation: T1 < timestamp <= T2; test: timestamp > T2",
                  "boundaries": {"T1": data.t1, "T2": data.t2}, **split},
        "classification": {
            "question": "预测 T1 之后用户对电影下一次评分是否不低于 4 分",
            "feature_names": FEATURE_NAMES,
            "history_policy": "训练行只用严格更早时间戳的评分；同时间戳事件互不可见。验证、测试和线上查询只用 T1 冻结历史。",
            "primary_metrics": ["f1", "pr_auc"],
            "baseline": classification_baseline,
            "algorithms": classification,
        },
        "clustering": {**profile_summary, "algorithms": clustering,
                       "metrics_scope": "训练期画像；轮廓系数使用固定随机种子的最多 1200 用户样本"},
        "reduction": {"original_dimensions": len(profile_summary["feature_names"]),
                      "algorithms": reduction,
                      "comparison_note": "压缩空间轮廓系数与原空间值因距离结构不同不能直接视为信息保留；同时给出压缩标签在原空间的轮廓系数与 ARI。"},
        "model_bundle_version": model_digest,
        "dependencies": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                         "joblib": joblib.__version__},
        "limitations": [
            "高分是评分 >= 4 的代理标签，不是对未来真实偏好的因果判断。",
            "训练期未出现的用户或电影可触发冷启动特征；在线查询拒绝训练期已出现的用户电影对。",
            "群体标签只是训练期画像的无监督划分，编号不具有跨算法含义，也不是用户固有属性。",
            "群体代表电影取自训练期高分记录，反映频次而不证明因果偏好。",
            "二维图仅展示前两维；高维坐标和训练期数值画像完整保存在产物中，聚类对照使用全部训练维度。",
            "参数变化需要创建新的分析任务，现有已发布模型不会即时重训。",
            "聚类和降维课程 PPT 尚待核对；当前 KMeans、Agglomerative、PCA、NMF 为常见候选实现。",
        ],
    }
    scaler = profile_models["profile_scaler"]
    _write_json(output_dir / "profiles.json", {"scope": "train_only", "T1": data.t1,
                                                "feature_names": profile_summary["feature_names"],
                                                "scaler": {"method": "MinMaxScaler",
                                                           "fitted_on": "eligible_train_users_only",
                                                           "data_min": scaler.data_min_.tolist(),
                                                           "data_max": scaler.data_max_.tolist(),
                                                           "scale": scaler.scale_.tolist(),
                                                           "offset": scaler.min_.tolist()},
                                                "users": profiles,
                                                "known_users_without_profile": [uid for uid in sorted(data.users)
                                                                                if train_counts[uid] < profile_summary["min_train_ratings"]]})
    _write_json(output_dir / "clusters.json", {"algorithms": clustering,
                                                "eligible_users": profile_summary["eligible_users"]})
    _write_json(output_dir / "coordinates.json", {"original_dimensions": report["reduction"]["original_dimensions"],
                                                   "users": coordinates})
    _write_json(output_dir / "report.json", report)
    manifest = {
        "schema_version": 1,
        "tool_id": "analysis.iteration2",
        "analysis_version": analysis_version,
        "model_bundle_version": model_digest,
        "source": source,
        "configuration": config,
        "created_at_utc": report["created_at_utc"],
        "artifacts": [{"name": name, "bytes": (output_dir / name).stat().st_size,
                       "sha256": file_sha256(output_dir / name)} for name in ARTIFACTS],
    }
    _write_json(output_dir / "manifest.json", manifest)
    if on_progress:
        on_progress("completed", "六个分析工具与版本化产物已生成")
    return report


def verify_analysis_manifest(output_dir) -> dict:
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("迭代二产物清单缺失，分析尚未完成。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("tool_id") != "analysis.iteration2":
        raise ValueError("迭代二产物清单工具标识无效。")
    names = {item.get("name") for item in manifest.get("artifacts", [])}
    if names != set(ARTIFACTS):
        raise ValueError("迭代二产物清单文件列表不完整。")
    for item in manifest["artifacts"]:
        artifact = output_dir / item["name"]
        if (not artifact.is_file() or artifact.stat().st_size != item.get("bytes")
                or file_sha256(artifact) != item.get("sha256")):
            raise ValueError(f"迭代二产物摘要不匹配：{item['name']}")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    if (report.get("status") != "completed"
            or report.get("analysis_version") != manifest.get("analysis_version")
            or report.get("source") != manifest.get("source")
            or report.get("configuration") != manifest.get("configuration")
            or report.get("model_bundle_version") != manifest.get("model_bundle_version")):
        raise ValueError("迭代二报告与清单的状态或版本不一致。")
    source = manifest["source"]
    source_dir = Path(source["source_dir"])
    if not (source_dir / "manifest.json").is_file():
        raise FileNotFoundError("迭代一源清单已不存在；无法核验分析结果来源。")
    if file_sha256(source_dir / "manifest.json") != source["source_manifest_sha256"]:
        raise ValueError("迭代一源清单自分析后发生变化。")
    current_source = verify_manifest(source_dir)
    if (current_source.get("clean_data_version") != source["clean_data_version"]
            or current_source.get("time_boundaries", {}).get("T1") != source["T1"]
            or current_source.get("time_boundaries", {}).get("T2") != source["T2"]):
        raise ValueError("迭代一源数据版本或时间边界已变化。")
    return manifest


def load_report(output_dir=DEFAULT_OUTPUT) -> dict:
    verify_analysis_manifest(output_dir)
    return json.loads((Path(output_dir) / "report.json").read_text(encoding="utf-8"))


def _load_published(output_dir, with_models=False):
    try:
        report = load_report(output_dir)
        models = joblib.load(Path(output_dir) / "models.joblib") if with_models else None
        return report, models, None
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return None, None, {"status": "unavailable", "reason": str(error)}


def _selected(category, algorithm):
    allowed = {"classification": ("logistic_regression", "decision_tree"),
               "clustering": ("kmeans", "agglomerative"),
               "reduction": ("pca", "nmf")}[category]
    if algorithm is None:
        return allowed, None
    if algorithm not in allowed:
        return (), {"status": "invalid_request", "reason": f"未登记的{category}算法：{algorithm}"}
    return (algorithm,), None


def predict_pair(user_id, movie_id, output_dir=DEFAULT_OUTPUT, algorithm=None) -> dict:
    selected, error = _selected("classification", algorithm)
    if error:
        return error
    report, models, error = _load_published(output_dir, with_models=True)
    if error:
        return error
    try:
        uid, mid = int(user_id), int(movie_id)
    except (TypeError, ValueError):
        return {"status": "invalid_request", "reason": "需要有效的 UserID 和 MovieID。"}
    if uid not in models["users"] or mid not in models["movies"]:
        return {"status": "unknown_id", "reason": "目标 UserID 或 MovieID 不存在于已校验的清洗数据。",
                "user_id": uid, "movie_id": mid}
    pair = np.uint64((uid << 32) | mid)
    training_pairs = models["known_training_pairs"]
    position = int(np.searchsorted(training_pairs, pair))
    if position < len(training_pairs) and training_pairs[position] == pair:
        return {"status": "known_training_pair", "reason": "此用户电影组合在 T1 前已有评分；为避免将已知事件包装成前瞻预测，本接口拒绝该查询。",
                "user_id": uid, "movie_id": mid, "as_of_utc": report["source"]["T1_utc"]}
    frozen = models["frozen"]
    row = pair_features(uid, mid, frozen)
    results = []
    for name in selected:
        detail = report["classification"]["algorithms"][name]
        probability = float(models["classifiers"][name].predict_proba(row)[0, 1])
        threshold = detail["validation"]["threshold"]
        results.append({"tool_id": detail["tool_id"], "algorithm": name,
                        "predicted_high_rating": bool(probability >= threshold),
                        "positive_probability": probability,
                        "decision_threshold": threshold,
                        "model_version": detail["model_version"],
                        "params": detail["params"], "test_metrics": detail["test"]})
    return {"status": "completed", "task": "classification", "user_id": uid,
            "movie_id": mid, "movie_title": models["movies"][mid]["title"],
            "as_of_utc": report["source"]["T1_utc"],
            "interpretation": "仅使用 T1 前历史，预测 T1 之后一次尚未发生的评分是否 >=4；不是实际评分。",
            "cold_start": {"user": int(frozen["user_count"][uid]) == 0,
                           "movie": int(frozen["movie_count"][mid]) == 0},
            "results": results}


def get_user_clusters(user_id=None, output_dir=DEFAULT_OUTPUT, algorithm=None) -> dict:
    selected, error = _selected("clustering", algorithm)
    if error:
        return error
    report, models, error = _load_published(output_dir, with_models=True)
    if error:
        return error
    profile_data = json.loads((Path(output_dir) / "profiles.json").read_text(encoding="utf-8"))
    if user_id is None:
        results = [{"tool_id": f"clustering.{name}", "algorithm": name,
                    **report["clustering"]["algorithms"][name]}
                   for name in selected]
        return {"status": "completed", "task": "clustering", "scope": "all_eligible_train_users",
                "eligible_users": report["clustering"]["eligible_users"], "results": results}
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return {"status": "invalid_request", "reason": "需要有效的 UserID。"}
    if uid not in models["users"]:
        return {"status": "unknown_id", "reason": "目标 UserID 不存在于已校验的清洗数据。",
                "user_id": uid}
    profiles = {row["user_id"]: row for row in profile_data["users"]}
    if uid not in profiles:
        return {"status": "insufficient_history", "reason": "该用户在 T1 前不足 20 条评分，未分配群体。",
                "user_id": uid, "train_ratings": int(models["frozen"]["user_count"][uid])}
    profile = profiles[uid]
    results = []
    for name in selected:
        detail = report["clustering"]["algorithms"][name]
        label = profile["clusters"][name]
        results.append({"tool_id": detail["tool_id"], "algorithm": name,
                        "cluster_id": label,
                        "is_noise": False,
                        "segment": next(group for group in detail["segments"] if group["cluster_id"] == label),
                        "model_version": detail["model_version"],
                        "params": detail["params"], "metrics": detail["metrics"]})
    return {"status": "completed", "task": "clustering", "user_id": uid,
            "profile": profile, "as_of_utc": report["source"]["T1_utc"], "results": results}


def get_reduction(output_dir=DEFAULT_OUTPUT, algorithm=None, limit=300, user_ids=None) -> dict:
    selected, error = _selected("reduction", algorithm)
    if error:
        return error
    report, _, error = _load_published(output_dir)
    if error:
        return error
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"status": "invalid_request", "reason": "limit 必须是正整数。"}
    if not 1 <= limit <= 6000:
        return {"status": "invalid_request", "reason": "limit 必须介于 1 和 6000。"}
    coordinate_data = json.loads((Path(output_dir) / "coordinates.json").read_text(encoding="utf-8"))
    rows = coordinate_data["users"]
    unavailable_users = []
    if user_ids is not None:
        if not isinstance(user_ids, (list, tuple)) or not 1 <= len(user_ids) <= 20:
            return {"status": "invalid_request", "reason": "user_ids 必须包含 1 到 20 个 UserID。"}
        try:
            requested = list(dict.fromkeys(int(uid) for uid in user_ids))
        except (TypeError, ValueError):
            return {"status": "invalid_request", "reason": "user_ids 必须是有效整数。"}
        by_id = {row["user_id"]: row for row in rows}
        known = json.loads((Path(output_dir) / "profiles.json").read_text(encoding="utf-8"))
        insufficient = set(known["known_users_without_profile"])
        rows = [by_id[uid] for uid in requested if uid in by_id]
        unavailable_users = [{"user_id": uid,
                              "status": "insufficient_history" if uid in insufficient else "unknown_id"}
                             for uid in requested if uid not in by_id]
    results = []
    for name in selected:
        detail = report["reduction"]["algorithms"][name]
        results.append({"tool_id": detail["tool_id"], "algorithm": name,
                        "model_version": detail["model_version"],
                        "original_dimensions": detail["original_dimensions"],
                        "actual_dimensions": detail["actual_dimensions"],
                        "coordinate_dimensions": detail.get("coordinate_dimensions", 2),
                        "information": detail["information"], "params": detail["params"],
                        "fit_seconds": detail["fit_seconds"],
                        "attempted_dimensions": detail.get("attempted_dimensions", [detail["actual_dimensions"]]),
                        "fit_iterations": detail.get("fit_iterations"),
                        "converged": detail.get("converged"),
                        "input_matrix_bytes": detail["input_matrix_bytes"],
                        "coordinate_matrix_bytes": detail["coordinate_matrix_bytes"],
                        "clustering_comparison": detail["clustering_comparison"],
                        "total_coordinates": len(coordinate_data["users"]),
                        "coordinates": [{"user_id": row["user_id"], "x": row[name][0], "y": row[name][1],
                                         "vector": row[name],
                                         "clusters": row["clusters"]} for row in rows[:limit]]})
    status = ("completed" if rows else unavailable_users[0]["status"] if unavailable_users
              else "unavailable")
    return {"status": status, "task": "reduction", "results": results,
            "unavailable_users": unavailable_users}


def execute_analysis_tool(tool_id, output_dir=DEFAULT_OUTPUT, user_id=None, movie_id=None,
                          limit=300, user_ids=None) -> dict:
    """Explicit one-algorithm dispatch used by the Agent tool layer."""
    if tool_id not in ANALYSIS_TOOLS:
        return {"status": "invalid_request", "reason": f"未登记的分析工具：{tool_id}"}
    category, algorithm = ANALYSIS_TOOLS[tool_id]
    if category == "classification":
        return predict_pair(user_id, movie_id, output_dir, algorithm)
    if category == "clustering":
        return get_user_clusters(user_id, output_dir, algorithm)
    return get_reduction(output_dir, algorithm, limit, user_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verified MovieLens iteration-two analysis")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", type=Path, help="Verify an existing iteration-two output directory")
    parser.add_argument("--n-clusters", type=int, help="Number of groups, 2 to 10 (default 5)")
    parser.add_argument("--target-dimensions", type=int,
                        help="Reduction dimensions, 2 to 10 (default 2)")
    parser.add_argument("--information-threshold", type=float,
                        help="Minimum centered reconstruction R², 0.5 to <1; exclusive with dimensions")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_analysis_manifest(args.verify)
        print(json.dumps({"status": "verified", "analysis_version": manifest["analysis_version"],
                          "artifacts": len(manifest["artifacts"])}, ensure_ascii=False))
    else:
        config = {key: value for key, value in {
            "n_clusters": args.n_clusters,
            "target_dimensions": args.target_dimensions,
            "information_threshold": args.information_threshold,
        }.items() if value is not None}
        result = run_analysis(args.source, args.output,
                              on_progress=lambda stage, message: print(f"[{stage}] {message}", flush=True),
                              config=config)
        print(json.dumps({"status": result["status"], "analysis_version": result["analysis_version"],
                          "output": str(args.output)}, ensure_ascii=False))
