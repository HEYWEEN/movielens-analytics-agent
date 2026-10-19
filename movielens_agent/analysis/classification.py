"""Leakage-aware high-rating classification, with a frozen T1 inference state."""

from __future__ import annotations

import math
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .common import Dataset, GENRES

HISTORY_START = 22
FEATURE_NAMES = ["gender_male", "age_scaled", "occupation_scaled", "release_year_scaled"] + [
    f"genre_{name}" for name in GENRES
] + ["user_prior_count_log", "user_prior_mean", "user_prior_high_rate",
     "movie_prior_count_log", "movie_prior_mean", "movie_prior_high_rate",
     "user_has_history", "movie_has_history"]


def _static_arrays(data: Dataset):
    user_static = np.zeros((max(data.users) + 1, 3), dtype=np.float32)
    for uid, (gender, age, occupation) in data.users.items():
        user_static[uid] = (float(gender == "M"), age / 56, occupation / 20)
    movie_static = np.zeros((max(data.movies) + 1, 1 + len(GENRES)), dtype=np.float32)
    for mid, info in data.movies.items():
        year = info["year"]
        movie_static[mid, 0] = (year - 1900) / 110 if year else 0
        movie_static[mid, 1:] = info["flags"]
    return user_static, movie_static


def _history_features(uc, us, uh, mc, ms, mh):
    """Fixed priors avoid using even aggregate future labels in a training row."""
    return (math.log1p(uc), (us + 4 * 3.5) / (uc + 4) / 5,
            (uh + 4 * 0.55) / (uc + 4),
            math.log1p(mc), (ms + 10 * 3.5) / (mc + 10) / 5,
            (mh + 10 * 0.55) / (mc + 10), float(uc > 0), float(mc > 0))


def build_features(data: Dataset):
    """Training rows see strictly earlier timestamps; later rows see frozen T1 history.

    Rows sharing a timestamp are evaluated before any of them update history. Validation
    and test labels never update the state used by another feature row.
    """
    user_static, movie_static = _static_arrays(data)
    n = len(data.ratings)
    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, :3] = user_static[data.user_ids]
    features[:, 3:HISTORY_START] = movie_static[data.movie_ids]
    user_count = np.zeros(len(user_static), dtype=np.int32)
    user_sum = np.zeros(len(user_static), dtype=np.float64)
    user_high = np.zeros(len(user_static), dtype=np.int32)
    movie_count = np.zeros(len(movie_static), dtype=np.int32)
    movie_sum = np.zeros(len(movie_static), dtype=np.float64)
    movie_high = np.zeros(len(movie_static), dtype=np.int32)

    ordered = data.split["train"][np.argsort(data.timestamps[data.split["train"]], kind="stable")]
    pending = []
    previous_time = None
    for index in ordered:
        timestamp = int(data.timestamps[index])
        if timestamp != previous_time:
            for old in pending:
                uid, mid, rating = int(data.user_ids[old]), int(data.movie_ids[old]), int(data.ratings[old])
                user_count[uid] += 1; user_sum[uid] += rating; user_high[uid] += rating >= 4
                movie_count[mid] += 1; movie_sum[mid] += rating; movie_high[mid] += rating >= 4
            pending.clear()
            previous_time = timestamp
        uid, mid = int(data.user_ids[index]), int(data.movie_ids[index])
        features[index, HISTORY_START:] = _history_features(
            user_count[uid], user_sum[uid], user_high[uid],
            movie_count[mid], movie_sum[mid], movie_high[mid])
        pending.append(index)
    for old in pending:
        uid, mid, rating = int(data.user_ids[old]), int(data.movie_ids[old]), int(data.ratings[old])
        user_count[uid] += 1; user_sum[uid] += rating; user_high[uid] += rating >= 4
        movie_count[mid] += 1; movie_sum[mid] += rating; movie_high[mid] += rating >= 4
    for name in ("validation", "test"):
        for index in data.split[name]:
            uid, mid = int(data.user_ids[index]), int(data.movie_ids[index])
            features[index, HISTORY_START:] = _history_features(
                user_count[uid], user_sum[uid], user_high[uid],
                movie_count[mid], movie_sum[mid], movie_high[mid])
    frozen = {
        "user_static": user_static, "movie_static": movie_static,
        "user_count": user_count, "user_sum": user_sum, "user_high": user_high,
        "movie_count": movie_count, "movie_sum": movie_sum, "movie_high": movie_high,
    }
    return features, frozen


def pair_features(user_id: int, movie_id: int, frozen: dict) -> np.ndarray:
    row = np.empty((1, len(FEATURE_NAMES)), dtype=np.float32)
    row[0, :3] = frozen["user_static"][user_id]
    row[0, 3:HISTORY_START] = frozen["movie_static"][movie_id]
    row[0, HISTORY_START:] = _history_features(
        frozen["user_count"][user_id], frozen["user_sum"][user_id], frozen["user_high"][user_id],
        frozen["movie_count"][movie_id], frozen["movie_sum"][movie_id], frozen["movie_high"][movie_id])
    return row


def _metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    metrics = {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "rows": int(len(labels)),
    }
    return metrics


def train_classifiers(data: Dataset, features: np.ndarray, on_progress=None):
    y = (data.ratings >= 4).astype(np.uint8)
    train, validation, test = (data.split[name] for name in ("train", "validation", "test"))
    choices = {
        "logistic_regression": [
            ({"C": c, "max_iter": 250, "solver": "lbfgs", "random_state": 42},
             make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=250,
                                                                 solver="lbfgs", random_state=42)))
            for c in (0.3, 1.0)
        ],
        "decision_tree": [
            ({"max_depth": depth, "min_samples_leaf": 50, "random_state": 42},
             DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=42))
            for depth in (8, 12)
        ],
    }
    baseline_probability = float(np.mean(y[train]))
    baseline = {
        "name": "train_positive_rate_constant",
        "train_positive_rate": baseline_probability,
        "test": _metrics(y[test], np.full(len(test), baseline_probability), 0.5),
        "note": "始终输出训练集正类率作为概率；由于其大于 0.5，测试期全部预测高分。",
    }
    results, trained = {}, {}
    for algorithm, candidates in choices.items():
        ranked = []
        for params, estimator in candidates:
            if on_progress:
                on_progress("classification", f"正在训练 {algorithm}，参数 {params}")
            began = time.perf_counter()
            estimator.fit(features[train], y[train])
            fit_seconds = time.perf_counter() - began
            validation_prob = estimator.predict_proba(features[validation])[:, 1]
            thresholds = np.arange(0.30, 0.701, 0.05)
            threshold = max(thresholds, key=lambda t: (
                f1_score(y[validation], validation_prob >= t), -abs(t - 0.5)))
            metrics = _metrics(y[validation], validation_prob, threshold)
            ranked.append((metrics["pr_auc"], metrics["f1"], params, estimator,
                           fit_seconds, metrics, threshold))
        best = max(ranked, key=lambda item: (item[0], item[1]))
        _, _, params, estimator, fit_seconds, validation_metrics, threshold = best
        test_prob = estimator.predict_proba(features[test])[:, 1]
        results[algorithm] = {
            "params": params,
            "validation": validation_metrics,
            "test": _metrics(y[test], test_prob, threshold),
            "candidate_validation": [{"params": item[2], "metrics": item[5]}
                                     for item in ranked],
            "fit_seconds": round(fit_seconds, 3),
            "serialized_model_bytes": len(pickle.dumps(estimator)),
            "selection_rule": "验证集 PR-AUC 最大；决策阈值在验证集最大化 F1；测试集仅用于选定后的最终评价",
            "positive_class": "评分 >= 4",
        }
        trained[algorithm] = estimator
    return results, trained, baseline
