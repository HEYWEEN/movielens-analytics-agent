"""Verify an iteration-one run from its published artifacts."""

import argparse
import json
from pathlib import Path

from movielens_agent.storage.artifacts import verify_manifest


TABLES = ("users", "movies", "ratings")
DIMENSIONS = ("Accurate", "Complete", "Unique", "Up-to-date", "Consistent")


def verify_run(run_dir: Path, require_hadoop: bool = True) -> dict:
    manifest = verify_manifest(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if manifest["task_id"] != report["task_id"] or run_dir.name != report["task_id"]:
        raise ValueError("任务 ID 与产物清单不一致")
    for key in ("raw_data_version", "clean_data_version", "rule_version", "score_config_version"):
        if manifest[key] != report[key]:
            raise ValueError(f"产物清单中的 {key} 与报告不一致")
    if manifest["time_boundaries"] != report["time_boundaries"]:
        raise ValueError("时间边界与产物清单不一致")
    for phase in ("raw", "clean"):
        if set(report[phase]["scores"]) != set(DIMENSIONS):
            raise ValueError(f"{phase} 阶段五维评分不完整")
        if set(report[phase]["counts"]) != set(TABLES):
            raise ValueError(f"{phase} 阶段数据量不完整")
    for table in TABLES:
        expected = report["clean"]["counts"][table]["rows"]
        actual = sum(1 for _ in (run_dir / f"{table}_clean.dat").open(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"{table} 清洗文件行数 {actual} 与报告 {expected} 不一致")
    disposition = report["disposition"]
    for table in TABLES:
        removed = report["raw"]["counts"][table]["rows"] - report["clean"]["counts"][table]["rows"]
        if disposition["removed_rows"][table] != removed:
            raise ValueError(f"{table} 数据量变化与处置统计不一致")
    if require_hadoop:
        if report.get("engine") != "hadoop-streaming":
            raise ValueError("这次运行不是 Hadoop Streaming 结果")
        jobs = report.get("phase_jobs", {})
        if set(jobs) != {"before", "clean", "after"}:
            raise ValueError("缺少 Hadoop 三阶段作业证据")
        for phase in jobs:
            if not jobs[phase].get("output_path") or not jobs[phase].get("job_id"):
                raise ValueError(f"{phase} 缺少 Hadoop 作业 ID 或输出路径")
        before = json.loads((run_dir / "before_report.json").read_text(encoding="utf-8"))
        after = json.loads((run_dir / "after_report.json").read_text(encoding="utf-8"))
        if before["raw"]["scores"] != report["raw"]["scores"]:
            raise ValueError("清洗前独立评分与报告不一致")
        if after["raw"]["scores"] != report["clean"]["scores"]:
            raise ValueError("清洗后独立评分与报告不一致")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-local", action="store_true", help="仅核验本地验证产物")
    args = parser.parse_args()
    result = verify_run(args.run_dir, require_hadoop=not args.allow_local)
    print(json.dumps({
        "task_id": result["task_id"], "engine": result["engine"],
        "raw_scores": result["raw"]["scores"],
        "clean_scores": result["clean"]["scores"],
        "clean_data_version": result["clean_data_version"],
    }, ensure_ascii=False, indent=2))
