"""MovieLens 1M cleaning and scoring core; used by local and Hadoop runners."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RULE_VERSION = "ml1m-rules-v4"
SCORE_CONFIG_VERSION = "ml1m-quality-v2"
SNAPSHOT_EXCLUSIVE = datetime(2003, 3, 1, tzinfo=timezone.utc)
MIN_TIMESTAMP = 946684800  # 2000-01-01 UTC
MAX_TIMESTAMP = int(SNAPSHOT_EXCLUSIVE.timestamp()) - 1
FRESH_FROM = int(SNAPSHOT_EXCLUSIVE.timestamp()) - 365 * 86400
AGES = {1, 18, 25, 35, 45, 50, 56}
GENRES = {"Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"}
FIELDS = {"users": 5, "movies": 3, "ratings": 4}
QUALITY_METHOD = {
    "Accurate": {
        "formula": "(字段与取值合规记录数 - 无效外键评分数 - 同频冲突未核验行数) / 总记录数 × 100",
        "scope": "字段取值合理性、跨表引用及同频冲突的未核验状态",
        "limitation": "同频冲突仅作为未核验扣分，并不证明其余记录的真实程度。",
    },
    "Complete": {
        "formula": "(必需字段格数 - 空缺字段格数) / 必需字段格数 × 100",
        "scope": "三个文件规定的必需字段",
        "limitation": "不衡量数据集没有收集的属性。",
    },
    "Unique": {
        "formula": "(总记录数 - 重复业务键的多余记录数) / 总记录数 × 100",
        "scope": "用户/电影 ID 与评分事件键",
        "limitation": "无法解析业务键的畸形行不计入重复数。",
    },
    "Up-to-date": {
        "formula": "发布前最后 365 天内的有效评分数 / 有效时间评分数 × 100",
        "scope": "评分时间；固定参照为 2003-02-28 23:59:59 UTC",
        "limitation": "历史归档数据不应按今天的日期判定失效；用户和电影表无更新时间。",
    },
    "Consistent": {
        "formula": "(结构正确记录数 - 冲突记录数 - 无效外键评分数) / 总记录数 × 100",
        "scope": "字段结构、业务键冲突与跨表引用",
        "limitation": "结构一致不代表语义或真实世界信息正确。",
    },
}


def normalize(table: str, line: str):
    parts = [p.strip() for p in line.rstrip("\r\n").split("::")]
    missing = sum(not p for p in parts[:FIELDS[table]]) + max(0, FIELDS[table] - len(parts))
    if len(parts) != FIELDS[table]:
        return None, "field_count", missing, False
    if missing:
        return None, "missing_field", missing, True
    try:
        if table == "users":
            uid, gender, age, occupation, zipcode = parts
            if not (uid.isdecimal() and 1 <= int(uid) <= 6040 and gender in {"M", "F"}
                    and int(age) in AGES and 0 <= int(occupation) <= 20
                    and 1 <= len(zipcode) <= 10):
                raise ValueError
            parts = [str(int(uid)), gender, str(int(age)), str(int(occupation)), zipcode]
        elif table == "movies":
            mid, title, genres = parts
            if not (mid.isdecimal() and 1 <= int(mid) <= 3952 and title):
                raise ValueError
            genre_parts = [g.strip() for g in genres.split("|")]
            if not genre_parts or any(g not in GENRES for g in genre_parts):
                raise ValueError
            parts = [str(int(mid)), title, "|".join(dict.fromkeys(genre_parts))]
        else:
            uid, mid, rating, timestamp = parts
            if not (uid.isdecimal() and mid.isdecimal() and rating.isdecimal()
                    and 1 <= int(uid) <= 6040 and 1 <= int(mid) <= 3952
                    and 1 <= int(rating) <= 5
                    and timestamp.isdecimal()):
                raise ValueError
            ts = int(timestamp)
            if ts > 10**11 and ts % 1000 == 0:
                ts //= 1000  # exact millisecond timestamps are unambiguous
            if not MIN_TIMESTAMP <= ts <= MAX_TIMESTAMP:
                raise ValueError
            parts = [str(int(uid)), str(int(mid)), str(int(rating)), str(ts)]
    except ValueError:
        return None, "invalid_domain", missing, True
    return tuple(parts), None, missing, True


def score_components(stats: dict) -> dict:
    total = sum(stats[t]["rows"] for t in FIELDS)
    slots = sum(stats[t]["rows"] * FIELDS[t] for t in FIELDS)
    valid = sum(stats[t]["valid"] for t in FIELDS)
    duplicate = sum(stats[t]["duplicate"] for t in FIELDS)
    missing = sum(stats[t]["missing_slots"] for t in FIELDS)
    structural = sum(stats[t]["shape_ok"] for t in FIELDS)
    orphan = stats["ratings"]["orphan"]
    ambiguous = sum(stats[t]["ambiguous"] for t in FIELDS)
    conflicted = stats["users"]["conflict"] + stats["movies"]["conflict"] + stats["ratings"]["conflict"]
    timely = stats["ratings"]["fresh"]
    valid_times = stats["ratings"]["valid_time"]

    return {
        "Accurate": {"numerator": max(0, valid - orphan - ambiguous), "denominator": total},
        "Complete": {"numerator": slots - missing, "denominator": slots},
        "Unique": {"numerator": total - duplicate, "denominator": total},
        "Up-to-date": {"numerator": timely, "denominator": valid_times},
        "Consistent": {"numerator": max(0, structural - conflicted - orphan), "denominator": total},
    }


def score(stats: dict) -> dict:
    return {name: round(100 * item["numerator"] / item["denominator"], 3)
            if item["denominator"] else None
            for name, item in score_components(stats).items()}


def blank_stats():
    return {table: {key: 0 for key in ("rows", "valid", "shape_ok", "missing_slots", "normalized",
                                     "repaired_retained", "duplicate", "conflict", "ambiguous",
                                     "orphan", "fresh", "valid_time")}
            for table in FIELDS}


def process(records, emit):
    """Consume (table, raw line); emit (kind, payload) for clean/quarantine/report."""
    raw = blank_stats()
    clean = blank_stats()
    issues = Counter()
    entities = {"users": defaultdict(Counter), "movies": defaultdict(Counter)}
    normalized_candidates = {table: set() for table in FIELDS}
    unchanged_entities = {"users": set(), "movies": set()}
    examples_by_issue = defaultdict(list)
    inherited_uncertain = {table: set() for table in FIELDS}

    def quarantine(table, reason, line, count=1):
        issues[f"{table}.{reason}"] += count
        issue_key = f"{table}.{reason}"
        if len(examples_by_issue[issue_key]) < 2:
            examples_by_issue[issue_key].append({"table": table, "reason": reason,
                                                 "record": line[:240], "count": count})
        emit("quarantine", json.dumps({"table": table, "reason": reason,
                                        "record": line, "count": count}, ensure_ascii=False))

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as rating_spool:
        for table, line in records:
            if table == "uncertain":
                marker = json.loads(line)
                if marker.get("table") in FIELDS and isinstance(marker.get("business_key"), str):
                    inherited_uncertain[marker["table"]].add(marker["business_key"])
                continue
            if table not in FIELDS:
                continue
            if table == "ratings":
                rating_spool.write(line.rstrip("\r\n") + "\n")
                continue
            s = raw[table]
            s["rows"] += 1
            row, error, missing, shape = normalize(table, line)
            s["missing_slots"] += missing
            s["shape_ok"] += int(shape)
            if error:
                quarantine(table, error, line.rstrip("\r\n"))
                continue
            s["valid"] += 1
            normalized_line = "::".join(row)
            if normalized_line != line.rstrip("\r\n"):
                s["normalized"] += 1
                normalized_candidates[table].add(normalized_line)
            else:
                unchanged_entities[table].add(normalized_line)
            entities[table][row[0]][row] += 1

        accepted = {"users": set(), "movies": set()}
        for table in ("users", "movies"):
            for entity_id, variants in sorted(entities[table].items(), key=lambda item: int(item[0])):
                canonical = sorted(variants, key=lambda row: (-variants[row], row))[0]
                tied = len(variants) > 1 and sum(count == variants[canonical] for count in variants.values()) > 1
                uncertain = tied or entity_id in inherited_uncertain[table]
                if uncertain:
                    raw[table]["ambiguous"] += sum(variants.values())
                    clean[table]["ambiguous"] += 1
                if tied:
                    emit("uncertain", json.dumps({"table": table, "business_key": entity_id,
                                                   "selected": "::".join(canonical),
                                                   "variants": [{"record": "::".join(row), "count": count}
                                                                for row, count in sorted(variants.items())],
                                                   "reason": "equal_top_frequency_unverified"}, ensure_ascii=False))
                accepted[table].add(entity_id)
                for row, count in variants.items():
                    if row == canonical:
                        if count > 1:
                            raw[table]["duplicate"] += count - 1
                            quarantine(table, "duplicate", "::".join(row), count - 1)
                    else:
                        raw[table]["duplicate"] += count
                        raw[table]["conflict"] += count
                        quarantine(table, "conflicting_id", "::".join(row), count)
                emit(table, "::".join(canonical))
                clean[table]["rows"] += 1
                clean[table]["valid"] += 1
                clean[table]["shape_ok"] += 1
                canonical_line = "::".join(canonical)
                clean[table]["repaired_retained"] += int(
                    canonical_line in normalized_candidates[table]
                    and canonical_line not in unchanged_entities[table])

        rating_spool.seek(0)
        events = defaultdict(Counter)
        for line in rating_spool:
            s = raw["ratings"]
            s["rows"] += 1
            row, error, missing, shape = normalize("ratings", line)
            s["missing_slots"] += missing
            s["shape_ok"] += int(shape)
            if error:
                quarantine("ratings", error, line.rstrip("\r\n"))
                continue
            s["valid"] += 1
            if "::".join(row) != line.rstrip("\r\n"):
                s["normalized"] += 1
                normalized_candidates["ratings"].add("::".join(row))
            ts = int(row[3])
            s["valid_time"] += 1
            s["fresh"] += int(ts >= FRESH_FROM)
            if row[0] not in accepted["users"] or row[1] not in accepted["movies"]:
                s["orphan"] += 1
                quarantine("ratings", "orphan_reference", "::".join(row))
                continue
            events[(row[0], row[1], row[3])][row[2]] += 1

        # Only count a retained output as repaired if no identical raw line existed.
        rating_spool.seek(0)
        for line in rating_spool:
            normalized_candidates["ratings"].discard(line.rstrip("\r\n"))

        timestamps = []
        # Hadoop may feed files and partitions in a different order from local mode.
        # Canonical ordering keeps clean-file hashes and downstream data versions stable.
        for key in sorted(events, key=lambda value: tuple(int(part) for part in value)):
            variants = events[key]
            canonical_rating = sorted(variants, key=lambda rating: (-variants[rating], int(rating)))[0]
            business_key = "::".join(key)
            top_count = variants[canonical_rating]
            tied = len(variants) > 1 and sum(count == top_count for count in variants.values()) > 1
            uncertain = tied or business_key in inherited_uncertain["ratings"]
            if uncertain:
                raw["ratings"]["ambiguous"] += sum(variants.values())
                clean["ratings"]["ambiguous"] += 1
            if tied:
                emit("uncertain", json.dumps({"table": "ratings", "business_key": business_key,
                                               "selected": "::".join((*key[:2], canonical_rating, key[2])),
                                               "variants": [{"record": "::".join((*key[:2], rating, key[2])), "count": count}
                                                            for rating, count in sorted(variants.items())],
                                               "reason": "equal_top_frequency_unverified"}, ensure_ascii=False))
            for rating, count in variants.items():
                duplicate_count = count - 1 if rating == canonical_rating else count
                if duplicate_count:
                    raw["ratings"]["duplicate"] += duplicate_count
                    if rating != canonical_rating:
                        raw["ratings"]["conflict"] += count
                    quarantine("ratings", "duplicate" if rating == canonical_rating else "conflicting_event",
                               "::".join((key[0], key[1], rating, key[2])), duplicate_count)
            row = (key[0], key[1], canonical_rating, key[2])
            emit("ratings", "::".join(row))
            timestamps.append(int(key[2]))
            clean["ratings"]["rows"] += 1
            clean["ratings"]["valid"] += 1
            clean["ratings"]["shape_ok"] += 1
            clean["ratings"]["repaired_retained"] += int("::".join(row) in normalized_candidates["ratings"])
            clean["ratings"]["valid_time"] += 1
            clean["ratings"]["fresh"] += int(int(key[2]) >= FRESH_FROM)

    timestamps.sort()
    t1 = timestamps[min(len(timestamps) - 1, int(len(timestamps) * 0.8))] if timestamps else None
    t2 = timestamps[min(len(timestamps) - 1, int(len(timestamps) * 0.9))] if timestamps else None
    report = {
        "rule_version": RULE_VERSION,
        "snapshot_exclusive_utc": SNAPSHOT_EXCLUSIVE.isoformat(),
        "timestamp_limit_utc": datetime.fromtimestamp(MAX_TIMESTAMP, timezone.utc).isoformat(),
        "raw": {"counts": raw, "scores": score(raw)},
        "clean": {"counts": clean, "scores": score(clean)},
        "score_evidence": {"raw": score_components(raw), "clean": score_components(clean)},
        "issues": dict(sorted(issues.items())),
        "unresolved_ties": {
            "by_table": {table: clean[table]["ambiguous"] for table in FIELDS},
            "total_retained": sum(clean[table]["ambiguous"] for table in FIELDS),
            "evidence_file": "unresolved_conflicts.jsonl",
            "note": "同频冲突保留项仍未核验；此数属于保留行，不另加到隔离或修复数量。",
        },
        "normalized_records": {table: raw[table]["normalized"] for table in FIELDS},
        "examples": [example for issue_key in sorted(examples_by_issue)
                     for example in examples_by_issue[issue_key]],
        "quality_method": QUALITY_METHOD,
        "time_boundaries": {"T1": t1, "T2": t2,
                            "T1_utc": datetime.fromtimestamp(t1, timezone.utc).isoformat() if t1 else None,
                            "T2_utc": datetime.fromtimestamp(t2, timezone.utc).isoformat() if t2 else None},
        "limitations": [
            "Accurate 额外扣除同频冲突未核验行，但仍不能证明其余值真实。",
            "用户人口属性由用户自行填写，无法从本数据集核验。",
            "Up-to-date 固定以 2003-02-28 日终为参照，使用 365 天窗口，不按今天的日期打分。",
            "同频冲突仍按字典序选择一个供下游使用，同时标记为未核验；不是确认其内容正确。",
            "单个 Hadoop reducer 适合本次约百万条记录的实验，不适合任意规模。",
        ],
    }
    removed = {table: raw[table]["rows"] - clean[table]["rows"] for table in FIELDS}
    by_table = {}
    for table in FIELDS:
        issue_prefix = table + "."
        table_issues = {key.removeprefix(issue_prefix): count for key, count in issues.items()
                        if key.startswith(issue_prefix)}
        disposition = {
            "raw_rows": raw[table]["rows"],
            "retained_rows": clean[table]["rows"],
            "repaired_retained": clean[table]["repaired_retained"],
            "retained_unverified_ties": clean[table]["ambiguous"],
            "exact_duplicates": table_issues.get("duplicate", 0),
            "conflicting_rows": sum(count for key, count in table_issues.items() if key.startswith("conflicting_")),
            "invalid_or_orphan_rows": sum(count for key, count in table_issues.items()
                                          if key != "duplicate" and not key.startswith("conflicting_")),
        }
        if disposition["raw_rows"] != disposition["retained_rows"] + disposition["exact_duplicates"] + disposition["conflicting_rows"] + disposition["invalid_or_orphan_rows"]:
            raise AssertionError(f"Disposition does not reconcile for {table}")
        by_table[table] = disposition
    report["disposition"] = {
        "by_table": by_table,
        "removed_rows": removed,
        "repaired_retained": sum(clean[table]["repaired_retained"] for table in FIELDS),
        "retained_unverified_ties": sum(clean[table]["ambiguous"] for table in FIELDS),
        "exact_duplicates": sum(count for key, count in issues.items() if key.endswith(".duplicate")),
        "conflicting_rows": sum(count for key, count in issues.items() if ".conflicting_" in key),
        "invalid_or_orphan_rows": sum(count for key, count in issues.items()
                                      if not key.endswith(".duplicate") and ".conflicting_" not in key),
        "normalization_note": "normalized_records 统计去重前经过规范化的输入行，不等于最终保留的修复行数。",
    }
    report["interpretation"] = {
        "Accurate": "字段或取值不合规、跨表引用无效的行被隔离；同频冲突保留项仍未核验并在此维度扣分。",
        "Complete": "缺失必需字段的行被隔离，未通过填充值来修复完整性。",
        "Unique": "完全重复的业务记录被去重，同一业务键的冲突变体被隔离。",
        "Up-to-date": "没有把旧时间戳改成新时间；分数变化来自保留评分的组成变化。",
        "Consistent": "结构错误与业务键冲突的行被隔离；通过检查并不证明语义真实。",
        "unresolved": [
            "人口属性、标题、类型和评分的真实值无法从这些文件独立核验。",
            "同频冲突的保留项仍未核验，字典序只是使输出稳定，不能证明其正确。",
            "历史时效评分只针对归档时点，不代表数据在今天仍然新。",
        ],
    }
    emit("report", json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return report


def version_for_files(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for table in sorted(paths):
        digest.update(table.encode())
        with paths[table].open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]
