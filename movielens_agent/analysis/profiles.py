"""Train-only user preference profiles, clustering and 2-D reductions."""

from __future__ import annotations

import pickle
import time
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .common import Dataset, GENRES, MIN_PROFILE_RATINGS

PROFILE_FEATURES = ([f"{genre}_watch_share" for genre in GENRES]
                    + [f"{genre}_smoothed_mean" for genre in GENRES]
                    + ["overall_high_rate"])


def build_profiles(data: Dataset):
    """Build every profile exclusively from ratings at or before T1."""
    max_uid = max(data.users)
    count = np.zeros(max_uid + 1, dtype=np.int32)
    high = np.zeros(max_uid + 1, dtype=np.int32)
    genre_count = np.zeros((max_uid + 1, len(GENRES)), dtype=np.int32)
    genre_sum = np.zeros((max_uid + 1, len(GENRES)), dtype=np.float64)
    for index in data.split["train"]:
        uid, mid, rating = (int(data.user_ids[index]), int(data.movie_ids[index]),
                            int(data.ratings[index]))
        count[uid] += 1
        high[uid] += rating >= 4
        genre_indexes = np.flatnonzero(data.movies[mid]["flags"])
        genre_count[uid, genre_indexes] += 1
        genre_sum[uid, genre_indexes] += rating
    eligible = np.asarray([uid for uid in sorted(data.users) if count[uid] >= MIN_PROFILE_RATINGS],
                          dtype=np.int32)
    if len(eligible) < 10:
        raise ValueError(f"训练期满足至少 {MIN_PROFILE_RATINGS} 条历史评分的用户不足。")
    watch_share = genre_count[eligible] / count[eligible, None]
    mean = (genre_sum[eligible] + 5 * 3.5) / (genre_count[eligible] + 5) / 5
    positive = high[eligible] / count[eligible]
    raw = np.column_stack((watch_share, mean, positive)).astype(np.float32)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(raw).astype(np.float32)
    rows = []
    for row_index, uid in enumerate(eligible):
        watched = np.argsort(-genre_count[uid], kind="stable")[:5]
        rows.append({
            "user_id": int(uid), "train_ratings": int(count[uid]),
            "high_rate": round(float(positive[row_index]), 4),
            "raw_feature_vector": [float(value) for value in raw[row_index]],
            "feature_vector": [float(value) for value in features[row_index]],
            "top_genres": [{"name": GENRES[g], "watched": int(genre_count[uid, g]),
                            "mean_rating": round(float(genre_sum[uid, g] / genre_count[uid, g]), 3)}
                           for g in watched if genre_count[uid, g] > 0],
        })
    return eligible, features, rows, scaler, count


def _cluster_fit(name, features, k=5):
    if name == "kmeans":
        estimator = KMeans(n_clusters=k, n_init=10, random_state=42)
    elif name == "agglomerative":
        estimator = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        raise ValueError(f"Unknown clustering algorithm: {name}")
    began = time.perf_counter()
    labels = estimator.fit_predict(features)
    return estimator, labels, time.perf_counter() - began


def _quality(features, labels, sample_indexes):
    if len(np.unique(labels)) < 2:
        return {"silhouette": None, "calinski_harabasz": None,
                "davies_bouldin": None, "silhouette_sample_users": int(len(sample_indexes))}
    sample_labels = labels[sample_indexes]
    silhouette = (float(silhouette_score(features[sample_indexes], sample_labels))
                  if len(np.unique(sample_labels)) > 1 else None)
    return {
        "silhouette": silhouette,
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "silhouette_sample_users": int(len(sample_indexes)),
    }


def _segment_summaries(data: Dataset, eligible, labels, watch_share):
    uid_to_cluster = {int(uid): int(label) for uid, label in zip(eligible, labels)}
    overall_share = watch_share.mean(axis=0)
    movie_counts = [Counter() for _ in range(int(labels.max()) + 1)]
    global_movie_counts = Counter()
    for index in data.split["train"]:
        uid, mid, rating = int(data.user_ids[index]), int(data.movie_ids[index]), int(data.ratings[index])
        group = uid_to_cluster.get(uid)
        if group is None:
            continue
        if rating >= 4:
            movie_counts[group][mid] += 1
            global_movie_counts[mid] += 1
    segments = []
    for group in range(int(labels.max()) + 1):
        member_indexes = np.flatnonzero(labels == group)
        cluster_share = watch_share[member_indexes].mean(axis=0)
        gaps = cluster_share - overall_share
        distinctive = np.argsort(-gaps, kind="stable")[:4]
        top_genres = [GENRES[g] for g in distinctive]
        genre_lifts = [{"name": GENRES[g],
                        "cluster_watch_share": round(float(cluster_share[g]), 4),
                        "overall_watch_share": round(float(overall_share[g]), 4),
                        "difference_percentage_points": round(float(gaps[g] * 100), 2)}
                       for g in distinctive]
        candidates = [(mid, count,
                       (count / len(member_indexes)) /
                       (global_movie_counts[mid] / len(eligible)))
                      for mid, count in movie_counts[group].items() if count >= 8]
        if candidates:
            top_movies = sorted(candidates, key=lambda item: (-item[2], -item[1], item[0]))[:4]
        else:
            top_movies = [(mid, count, None) for mid, count in
                          movie_counts[group].most_common(4)]
        segments.append({
            "cluster_id": group,
            "size": int(len(member_indexes)),
            "top_genres": top_genres,
            "genre_lifts": genre_lifts,
            "representative_movies": [{"movie_id": mid, "title": data.movies[mid]["title"],
                                       "train_high_ratings": count,
                                       "over_index_ratio": round(float(lift), 3) if lift is not None else None}
                                      for mid, count, lift in top_movies],
            "summary": (f"相对所有合格用户，本群 {top_genres[0]} 类型的观影记录占比"
                        f"{'高' if gaps[distinctive[0]] >= 0 else '低'} "
                        f"{abs(genre_lifts[0]['difference_percentage_points']):.2f} 个百分点；"
                        "仅描述训练期记录，不推断形成原因。"),
        })
    return segments


def _reduce(features, name, target_dimensions, information_threshold):
    original_energy = float(np.square(features).sum())
    centered_energy = float(np.square(features - features.mean(axis=0)).sum())
    dimensions = ([target_dimensions] if information_threshold is None
                  else range(2, min(10, features.shape[1]) + 1))
    attempted_dimensions = []
    total_seconds = 0.0
    for dimension in dimensions:
        model = (PCA(n_components=dimension, random_state=42) if name == "pca" else
                 NMF(n_components=dimension, init="nndsvda", max_iter=2000, random_state=42))
        began = time.perf_counter()
        coords = model.fit_transform(features)
        seconds = time.perf_counter() - began
        total_seconds += seconds
        attempted_dimensions.append(dimension)
        reconstruction = model.inverse_transform(coords)
        squared_error = float(np.square(features - reconstruction).sum())
        reconstruction_r2 = 1 - squared_error / centered_energy
        energy_ratio = 1 - squared_error / original_energy
        if information_threshold is None or reconstruction_r2 >= information_threshold:
            return model, coords, total_seconds, reconstruction_r2, energy_ratio, attempted_dimensions
    raise ValueError(f"{name} 在最多 10 维内未达到信息保留阈值 {information_threshold}。")


def fit_profiles(data: Dataset, on_progress=None, n_clusters=5,
                 target_dimensions=2, information_threshold=None):
    eligible, features, profile_rows, scaler, all_counts = build_profiles(data)
    sample = np.random.default_rng(42).choice(len(eligible),
                                               size=min(1200, len(eligible)), replace=False)
    cluster_models, cluster_labels, clustering = {}, {}, {}
    for name in ("kmeans", "agglomerative"):
        if on_progress:
            on_progress("clustering", f"正在对 {len(eligible)} 位训练期用户执行 {name}")
        model, labels, seconds = _cluster_fit(name, features, n_clusters)
        cluster_models[name] = model
        cluster_labels[name] = labels
        clustering[name] = {
            "params": {"n_clusters": n_clusters, "linkage": "ward"} if name == "agglomerative"
                      else {"n_clusters": n_clusters, "n_init": 10, "random_state": 42},
            "metrics": _quality(features, labels, sample),
            "fit_seconds": round(seconds, 3),
            "serialized_model_bytes": len(pickle.dumps(model)),
            "segments": _segment_summaries(data, eligible, labels,
                                            np.asarray([row["raw_feature_vector"][:len(GENRES)]
                                                        for row in profile_rows], dtype=np.float32)),
        }
    for index, row in enumerate(profile_rows):
        row["clusters"] = {name: int(labels[index]) for name, labels in cluster_labels.items()}

    reduction_models, reductions, coordinates = {}, {}, {}
    for name in ("pca", "nmf"):
        if on_progress:
            on_progress("reduction", f"正在执行 {name} 降维与聚类对照")
        model, coords, seconds, reconstruction_r2, energy_ratio, attempted_dimensions = _reduce(
            features, name, target_dimensions, information_threshold)
        scaled = StandardScaler().fit_transform(coords).astype(np.float32)
        comparisons = {}
        for cluster_name in ("kmeans", "agglomerative"):
            _, reduced_labels, cluster_seconds = _cluster_fit(cluster_name, scaled, n_clusters)
            comparisons[cluster_name] = {
                "original_space": clustering[cluster_name]["metrics"],
                "reduced_space": _quality(scaled, reduced_labels, sample),
                "reduced_labels_original_space_silhouette": (
                    float(silhouette_score(features[sample], reduced_labels[sample]))
                    if len(np.unique(reduced_labels[sample])) > 1 else None),
                "adjusted_rand_vs_original": float(adjusted_rand_score(cluster_labels[cluster_name],
                                                                         reduced_labels)),
                "fit_seconds": round(cluster_seconds, 3),
            }
        coordinates[name] = coords.astype(np.float32)
        reduction_models[name] = model
        reductions[name] = {
            "params": {"n_components": int(coords.shape[1]), "random_state": 42,
                       **({"init": "nndsvda", "max_iter": 2000} if name == "nmf" else {})},
            "original_dimensions": int(features.shape[1]),
            "actual_dimensions": int(coords.shape[1]),
            "coordinate_dimensions": 2,
            "fit_iterations": int(getattr(model, "n_iter_", 1)),
            "converged": bool(getattr(model, "n_iter_", 1) < getattr(model, "max_iter", float("inf"))),
            "information": {"reconstruction_r2": reconstruction_r2,
                            "reconstruction_energy_ratio": energy_ratio,
                            "explained_variance_ratio_sum": (float(model.explained_variance_ratio_.sum())
                                                             if name == "pca" else None),
                            "requested_threshold": information_threshold,
                            "threshold_met": (reconstruction_r2 >= information_threshold
                                              if information_threshold is not None else None),
                            "note": "两算法共同以中心化重建 R²=1-SSE/SST 衡量保留信息并判断阈值；PCA 另报告解释方差。未中心化的能量比例不能代替解释方差。"},
            "fit_seconds": round(seconds, 3),
            "attempted_dimensions": attempted_dimensions,
            "input_matrix_bytes": int(features.nbytes),
            "coordinate_matrix_bytes": int(coords.nbytes),
            "serialized_model_bytes": len(pickle.dumps(model)),
            "clustering_comparison": comparisons,
        }
    coordinate_rows = [{"user_id": int(uid),
                        "pca": [float(v) for v in coordinates["pca"][i]],
                        "nmf": [float(v) for v in coordinates["nmf"][i]],
                        "clusters": profile_rows[i]["clusters"]}
                       for i, uid in enumerate(eligible)]
    summary = {
        "eligible_users": int(len(eligible)),
        "known_users_without_sufficient_train_history": int(len(data.users) - len(eligible)),
        "min_train_ratings": MIN_PROFILE_RATINGS,
        "feature_names": PROFILE_FEATURES,
        "profile_scope": "仅使用 timestamp <= T1 的评分；群体编号没有跨算法的语义对应关系",
    }
    models = {"profile_scaler": scaler, "cluster_models": cluster_models,
              "reduction_models": reduction_models}
    return summary, clustering, reductions, profile_rows, coordinate_rows, models, all_counts
