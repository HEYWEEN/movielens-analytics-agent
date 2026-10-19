"""Verified iteration-one input and shared, time-safe feature definitions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movielens_agent.storage.artifacts import verify_manifest

GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
GENRE_INDEX = {genre: index for index, genre in enumerate(GENRES)}
MIN_PROFILE_RATINGS = 20


@dataclass
class Dataset:
    source_dir: Path
    source_manifest: dict
    t1: int
    t2: int
    users: dict[int, tuple[str, int, int]]
    movies: dict[int, dict]
    user_ids: np.ndarray
    movie_ids: np.ndarray
    ratings: np.ndarray
    timestamps: np.ndarray
    split: dict[str, np.ndarray]


def load_dataset(source_dir: str | Path) -> Dataset:
    source_dir = Path(source_dir).resolve()
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("迭代一产物清单缺失；请先完成数据治理任务。")
    manifest = verify_manifest(source_dir)
    if manifest.get("tool_id") != "governance.clean_and_assess":
        raise ValueError("输入不是已登记的迭代一治理产物。")
    report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    if (report.get("clean_data_version") != manifest.get("clean_data_version")
            or report.get("time_boundaries") != manifest.get("time_boundaries")):
        raise ValueError("迭代一报告与产物清单的版本或时间边界不一致。")
    boundaries = manifest.get("time_boundaries") or {}
    t1, t2 = boundaries.get("T1"), boundaries.get("T2")
    if not (isinstance(t1, int) and isinstance(t2, int) and 0 < t1 < t2):
        raise ValueError("迭代一产物缺少有效的 T1/T2 时间边界。")

    users = {}
    for line in (source_dir / "users_clean.dat").open(encoding="utf-8"):
        uid, gender, age, occupation, _ = line.rstrip("\n").split("::")
        users[int(uid)] = (gender, int(age), int(occupation))
    movies = {}
    for line in (source_dir / "movies_clean.dat").open(encoding="utf-8"):
        mid, title, genres = line.rstrip("\n").split("::")
        flags = np.zeros(len(GENRES), dtype=np.float32)
        for genre in genres.split("|"):
            flags[GENRE_INDEX[genre]] = 1
        year_match = re.search(r"\((\d{4})\)\s*$", title)
        movies[int(mid)] = {
            "title": title,
            "genres": genres.split("|"),
            "flags": flags,
            "year": int(year_match.group(1)) if year_match else None,
        }
    uids, mids, values, times = [], [], [], []
    for line in (source_dir / "ratings_clean.dat").open(encoding="utf-8"):
        uid, mid, value, timestamp = line.rstrip("\n").split("::")
        uids.append(int(uid)); mids.append(int(mid))
        values.append(int(value)); times.append(int(timestamp))
    user_ids = np.asarray(uids, dtype=np.int32)
    movie_ids = np.asarray(mids, dtype=np.int32)
    ratings = np.asarray(values, dtype=np.uint8)
    timestamps = np.asarray(times, dtype=np.int64)
    if len(ratings) == 0:
        raise ValueError("清洗后的评分表为空，无法执行迭代二。")
    if any(uid not in users for uid in np.unique(user_ids)) or any(
            mid not in movies for mid in np.unique(movie_ids)):
        raise ValueError("评分表含有未登记的用户或电影 ID。")
    split = {
        "train": np.flatnonzero(timestamps <= t1),
        "validation": np.flatnonzero((timestamps > t1) & (timestamps <= t2)),
        "test": np.flatnonzero(timestamps > t2),
    }
    if min(len(indexes) for indexes in split.values()) == 0:
        raise ValueError("T1/T2 划分得到空训练、验证或测试集。")
    return Dataset(source_dir, manifest, t1, t2, users, movies,
                   user_ids, movie_ids, ratings, timestamps, split)


def split_summary(data: Dataset) -> dict:
    train_users = set(map(int, data.user_ids[data.split["train"]]))
    train_movies = set(map(int, data.movie_ids[data.split["train"]]))
    result = {}
    for name, indexes in data.split.items():
        users = set(map(int, data.user_ids[indexes]))
        movies = set(map(int, data.movie_ids[indexes]))
        result[name] = {
            "rows": int(len(indexes)),
            "positive": int(np.count_nonzero(data.ratings[indexes] >= 4)),
            "positive_rate": round(float(np.mean(data.ratings[indexes] >= 4)), 6),
            "unique_users": len(users), "unique_movies": len(movies),
            "users_without_train_history": len(users - train_users) if name != "train" else 0,
            "movies_without_train_history": len(movies - train_movies) if name != "train" else 0,
        }
    return result
