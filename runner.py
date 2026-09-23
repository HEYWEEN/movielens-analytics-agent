"""Run the same pipeline locally for development or as a real Hadoop job."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from pipeline import RULE_VERSION, SCORE_CONFIG_VERSION, process, version_for_files

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE.parent / "ml-1m"
DEFAULT_RUNS = BASE / "runs"
REGISTRY = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))


def resolve_configuration(data_version=None, rule_version=None, score_config_version=None,
                          data_dir_override=None):
    defaults = REGISTRY["defaults"]
    data_version = data_version or defaults["data_version"]
    rule_version = rule_version or defaults["rule_version"]
    score_config_version = score_config_version or defaults["score_config_version"]
    if data_version not in REGISTRY["data_versions"]:
        raise ValueError(f"未登记的数据版本：{data_version}")
    if rule_version not in REGISTRY["rule_versions"] or rule_version != RULE_VERSION:
        raise ValueError(f"未登记或不可执行的规则版本：{rule_version}")
    if score_config_version not in REGISTRY["score_config_versions"] or score_config_version != SCORE_CONFIG_VERSION:
        raise ValueError(f"未登记或不可执行的评分配置：{score_config_version}")
    data_dir = (Path(data_dir_override) if data_dir_override else
                (BASE / REGISTRY["data_versions"][data_version]["path"]).resolve())
    paths = inputs(data_dir)
    actual_version = version_for_files(paths)
    if actual_version != data_version:
        raise ValueError(f"数据版本不匹配：请求 {data_version}，文件实际为 {actual_version}")
    configuration = {"raw_data_version": data_version, "rule_version": rule_version,
                     "score_config_version": score_config_version}
    audit = REGISTRY["data_versions"][data_version].get("reference_audit")
    if audit and audit["rule_version"] == rule_version:
        configuration["reference_audit"] = audit
    return paths, configuration


def inputs(data_dir):
    paths = {table: data_dir / f"{table}.dat" for table in ("users", "movies", "ratings")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def save_output(lines, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=False)
    handles = {}
    names = {"users": "users_clean.dat", "movies": "movies_clean.dat",
             "ratings": "ratings_clean.dat", "quarantine": "quarantine.jsonl",
             "uncertain": "unresolved_conflicts.jsonl"}
    report = None
    try:
        for kind, payload in lines:
            if kind == "report":
                report = json.loads(payload)
                continue
            if kind not in names:
                raise ValueError(f"Unknown pipeline output: {kind}")
            if kind not in handles:
                handles[kind] = (output_dir / names[kind]).open("w", encoding="utf-8")
            handles[kind].write(payload + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    if report is None:
        raise RuntimeError("Pipeline produced no report")
    report.update(metadata)
    report["clean_data_version"] = f"{metadata['raw_data_version']}-{RULE_VERSION}"
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_local(data_dir=DEFAULT_DATA, runs_dir=DEFAULT_RUNS, task_id=None,
              data_version=None, rule_version=None, score_config_version=None, on_progress=None):
    data_dir, runs_dir = Path(data_dir), Path(runs_dir)
    paths, configuration = resolve_configuration(data_version, rule_version, score_config_version, data_dir)
    task_id = task_id or uuid.uuid4().hex[:12]

    def records():
        for table, path in paths.items():
            with path.open(encoding="latin-1") as source:
                for line in source:
                    yield table, line

    if on_progress:
        on_progress("local_pipeline", "正在本地检查、清洗和评分")
    # The pipeline streams output. A temporary JSONL file prevents buffering a million rows.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as spool:
        process(records(), lambda kind, payload: spool.write(json.dumps([kind, payload], ensure_ascii=False) + "\n"))
        spool.seek(0)
        lines = (json.loads(line) for line in spool)
        result = save_output(lines, runs_dir / task_id, {
            "task_id": task_id, "engine": "local-verification",
            **configuration,
        })
    if on_progress:
        on_progress("report", "本地报告已生成")
    return result


def run_hadoop(data_dir=DEFAULT_DATA, runs_dir=DEFAULT_RUNS, task_id=None,
               data_version=None, rule_version=None, score_config_version=None, on_progress=None):
    data_dir, runs_dir = Path(data_dir), Path(runs_dir)
    paths, configuration = resolve_configuration(data_version, rule_version, score_config_version, data_dir)
    jar = os.environ.get("HADOOP_STREAMING_JAR")
    if not jar or not Path(jar).is_file():
        raise RuntimeError("Set HADOOP_STREAMING_JAR to an existing hadoop-streaming JAR")
    if not all(shutil.which(command) for command in ("hadoop", "hdfs")):
        raise RuntimeError("hadoop and hdfs commands must be installed and on PATH")
    task_id = task_id or uuid.uuid4().hex[:12]
    standalone = os.environ.get("LAB2_HADOOP_LOCAL") == "1"
    hdfs_base = os.environ.get("LAB2_HDFS_BASE", "/tmp/lab2-movielens")
    job_base = f"{hdfs_base}/{task_id}"
    hdfs_input = f"{job_base}/raw-input"
    if on_progress:
        on_progress("prepare", "正在登记版本并上传原始数据")
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_input], check=True)
    for path in paths.values():
        subprocess.run(["hdfs", "dfs", "-put", str(path), hdfs_input + "/" + path.name], check=True)
    mapper = f"python3 {BASE / 'streaming.py'} mapper" if standalone else "python3 streaming.py mapper"

    def execute_phase(stage, source, phase):
        output = f"{job_base}/{stage}-output"
        label = {"before": "清洗前评分", "clean": "数据清洗", "after": "清洗后评分"}[stage]
        if on_progress:
            on_progress(stage, f"正在通过 Hadoop 执行{label}")
        reducer = (f"python3 {BASE / 'streaming.py'} reducer {phase}" if standalone else
                   f"python3 streaming.py reducer {phase}")
        command = ["hadoop", "jar", jar, "-D", "mapreduce.job.reduces=1"]
        if standalone:
            command.extend(["-D", "mapreduce.framework.name=local"])
        command.extend(["-files", f"{BASE / 'pipeline.py'},{BASE / 'streaming.py'}",
                        "-input", source, "-output", output,
                        "-mapper", mapper, "-reducer", reducer])
        job = subprocess.run(command, capture_output=True, text=True)
        if job.returncode:
            raise RuntimeError(f"{label}阶段失败：\n" + job.stderr[-6000:])
        match = re.search(r"Running job: (job_\S+)", job.stderr)
        completed = subprocess.run(["hdfs", "dfs", "-cat", output + "/part-*"],
                                   check=True, capture_output=True, text=True)
        lines = [line.split("\t", 1) for line in completed.stdout.splitlines()]
        if on_progress:
            on_progress(stage, f"{label}完成")
        return lines, {"job_id": match.group(1) if match else None, "output_path": output}

    def phase_report(lines, stage):
        reports = [json.loads(payload) for kind, payload in lines if kind == "report"]
        if len(reports) != 1:
            raise RuntimeError(f"{stage} 阶段未生成唯一的质量报告")
        return reports[0]

    before_lines, before_job = execute_phase("before", hdfs_input, "before")
    before_report = phase_report(before_lines, "before")
    clean_lines, clean_job = execute_phase("clean", hdfs_input, "clean")
    runs_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".staging-{task_id}-", dir=runs_dir) as staging:
        output_dir = Path(staging) / "outputs"
        report = save_output(clean_lines, output_dir, {
            "task_id": task_id, "engine": "hadoop-streaming",
            "hadoop_mode": "standalone-local" if standalone else "configured-cluster",
            "hadoop_job_id": clean_job["job_id"], "hdfs_output": clean_job["output_path"],
            **configuration,
        })
        del clean_lines
        clean_input = f"{job_base}/clean-input"
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", clean_input], check=True)
        for table in ("users", "movies", "ratings"):
            cleaned = output_dir / f"{table}_clean.dat"
            subprocess.run(["hdfs", "dfs", "-put", str(cleaned), clean_input + "/" + cleaned.name], check=True)
        uncertain = output_dir / "unresolved_conflicts.jsonl"
        if uncertain.is_file():
            subprocess.run(["hdfs", "dfs", "-put", str(uncertain),
                            clean_input + "/" + uncertain.name], check=True)
        after_lines, after_job = execute_phase("after", clean_input, "after")
        after_report = phase_report(after_lines, "after")
        if on_progress:
            on_progress("report", "正在核对三阶段结果并生成报告")
        if before_report["raw"]["scores"] != report["raw"]["scores"]:
            raise RuntimeError("清洗前评分与清洗作业中的原始数据评分不一致")
        if after_report["raw"]["scores"] != report["clean"]["scores"]:
            raise RuntimeError("清洗后独立评分与清洗作业中的结果不一致")
        if any(after_report["raw"]["counts"][table]["rows"] != report["clean"]["counts"][table]["rows"]
               for table in ("users", "movies", "ratings")):
            raise RuntimeError("清洗后独立评分的数据量与清洗作业不一致")
        report["phase_jobs"] = {"before": before_job, "clean": clean_job, "after": after_job}
        (output_dir / "before_report.json").write_text(json.dumps(before_report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "after_report.json").write_text(json.dumps(after_report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_dir.rename(runs_dir / task_id)
    if on_progress:
        on_progress("report", "三阶段报告已生成")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("local", "hadoop"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    report = (run_hadoop if args.engine == "hadoop" else run_local)(args.data_dir, args.runs_dir)
    print(json.dumps({"task_id": report["task_id"], "engine": report["engine"],
                      "scores": report["clean"]["scores"]}, ensure_ascii=False))
