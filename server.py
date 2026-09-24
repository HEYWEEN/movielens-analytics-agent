"""Small dependency-free task agent and web API for the iteration-one demo."""

import json
import os
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from runner import DEFAULT_DATA, DEFAULT_RUNS, REGISTRY, resolve_configuration, run_hadoop, run_local
from movielens_agent.storage.task_store import TaskStore
from movielens_agent.storage.artifacts import verify_manifest
from movielens_agent.tools.registry import get_tool

HERE = Path(__file__).resolve().parent
ENGINE = os.environ.get("LAB2_ENGINE", "local")
ENGINE_LABEL = "hadoop-standalone-local" if ENGINE == "hadoop" and os.environ.get("LAB2_HADOOP_LOCAL") == "1" else ENGINE
TASKS = {}
LOCK = threading.Lock()
TASK_STORE = TaskStore(DEFAULT_RUNS / "tasks.sqlite3")
RUN_SLOT = threading.BoundedSemaphore(1)


def latest_task():
    return TASK_STORE.latest_id()


def tool_run_pipeline(task_id, configuration):
    with RUN_SLOT:
        with LOCK:
            TASKS[task_id].update(status="running", progress="正在准备任务")
            TASK_STORE.update(task_id, status="running", progress="正在准备任务")
        try:
            get_tool("governance.clean_and_assess")
            report = (run_hadoop if ENGINE == "hadoop" else run_local)(
                DEFAULT_DATA, DEFAULT_RUNS, task_id, **configuration,
                on_progress=lambda stage, progress: set_progress(task_id, stage, progress))
            verify_manifest(DEFAULT_RUNS / task_id)
            with LOCK:
                TASKS[task_id].update(status="complete", progress="完成", report=report)
                TASK_STORE.update(task_id, status="complete", progress="完成")
        except Exception as exc:
            with LOCK:
                stage = TASKS[task_id].get("stage", "unknown")
                changes = {"status": "failed", "progress": f"{stage} 阶段失败",
                           "error": str(exc).splitlines()[0], "failed_stage": stage,
                           "detail": traceback.format_exc(limit=3)}
                TASKS[task_id].update(changes)
                TASK_STORE.update(task_id, **changes)


def tool_get_status(task_id):
    with LOCK:
        task = TASKS.get(task_id)
        if task:
            return {k: v for k, v in task.items() if k != "report"}
    stored = TASK_STORE.get(task_id)
    if stored:
        return stored
    report = tool_get_report(task_id)
    if not report:
        return None
    engine = ("hadoop-standalone-local" if report.get("hadoop_mode") == "standalone-local"
              else report["engine"])
    return {"task_id": task_id, "status": "complete", "progress": "完成", "engine": engine}


def set_progress(task_id, stage, progress):
    with LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(stage=stage, progress=progress)
            TASKS[task_id].setdefault("events", []).append({
                "stage": stage, "message": progress,
                "at_utc": datetime.now(timezone.utc).isoformat()})
            TASK_STORE.update(task_id, stage=stage, progress=progress, events=TASKS[task_id]["events"])


def tool_get_report(task_id):
    if not re.fullmatch(r"[0-9a-f]{12}", str(task_id)):
        return None
    with LOCK:
        task = TASKS.get(task_id)
        if task and task.get("report"):
            return task["report"]
    path = DEFAULT_RUNS / task_id / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def answer_from_report(question, report):
    raw, clean = report["raw"], report["clean"]
    issues = report["issues"]
    interpretation = report.get("interpretation", {})
    if any(word in question for word in ("未解决", "局限", "限制", "无法解决")):
        unresolved = interpretation.get("unresolved", report.get("limitations", []))
        answer = "目前不能解决或验证的问题：" + "；".join(item.rstrip("。；") for item in unresolved) + "。"
        audit = report.get("reference_audit")
        if audit:
            answer += (f"历史官方基准对照还发现用户 {audit['users_content_mismatched']} 条、"
                       f"电影 {audit['movies_content_mismatched']} 条内容不同；"
                       "这项对照不是每次任务重新计算的。")
        ties = report.get("unresolved_ties", {}).get("total_retained")
        if ties is not None:
            answer += f"本次同频冲突保留但未核验 {ties} 条，详见 unresolved_conflicts.jsonl。"
        return answer
    if any(word in question for word in ("时间", "T1", "T2", "划分")):
        time = report["time_boundaries"]
        return f"训练截止 T1：{time['T1_utc']}；验证截止 T2：{time['T2_utc']}。按清洗后评分时间的 80% 和 90% 分位确定，测试集位于 T2 之后。"
    if "评分" in question and any(word in question for word in ("依据", "公式", "怎么算")):
        methods = report.get("quality_method", {})
        evidence = report.get("score_evidence", {})
        return "五维评分口径与本次分子/分母（清洗前 → 后）：" + "；".join(
            f"{name}：{item['formula']}；"
            f"{evidence['raw'][name]['numerator']}/{evidence['raw'][name]['denominator']} → "
            f"{evidence['clean'][name]['numerator']}/{evidence['clean'][name]['denominator']}"
            if evidence else f"{name}：{item['formula']}"
            for name, item in methods.items())
    if any(word in question for word in ("依据", "规则", "如何清洗", "怎么清洗")):
        return (f"本次使用 {report['rule_version']}：按数据说明校验字段、ID 范围和取值；"
                "可确定的空格及毫秒时间戳做规范化；相同业务键去重，冲突值按频次及字典序选择；"
                "同频保留项标记为未核验，并在 Accurate 中扣分；"
                "无效字段和无效引用隔离。评分公式与限制见报告；它不能验证评分或人口属性的真实性。")
    if any(word in question for word in ("异常", "删除", "隔离", "去重", "修复")):
        top = sorted(issues.items(), key=lambda kv: -kv[1])[:8]
        detail = "；".join(f"{name}: {count}" for name, count in top)
        normalized = report["normalized_records"]
        disposition = report.get("disposition", {})
        return ("异常处置：" + (detail or "没有记录到异常") + "。去重 "
                + str(disposition.get("exact_duplicates", "未统计")) + " 条，冲突隔离 "
                + str(disposition.get("conflicting_rows", "未统计")) + " 条，其他无效或孤立记录隔离 "
                + str(disposition.get("invalid_or_orphan_rows", "未统计")) + " 条；规范化后最终保留 "
                + str(disposition.get("repaired_retained", "未统计")) + " 条；同频冲突保留但未核验 "
                + str(disposition.get("retained_unverified_ties", "未统计")) + " 条。规范化输入行："
                + "、".join(f"{table} {count}" for table, count in normalized.items())
                + " 条。规范化计数发生在去重前，不等于最终保留的修复行数。")
    if any(word in question for word in ("数量", "多少", "记录")):
        return "；".join(f"{table}：{raw['counts'][table]['rows']} → {clean['counts'][table]['rows']} 条"
                       for table in ("users", "movies", "ratings"))
    changes = []
    for name, before in raw["scores"].items():
        after = clean["scores"][name]
        change = f"{name} {before} → {after}" if before is not None else f"{name} 无法评价"
        explanation = interpretation.get(name, "")
        changes.append(change + (f"，{explanation.rstrip('。；')}" if explanation else ""))
    return "五维评分（清洗前 → 后，满分 100）：" + "；".join(changes) + "。"


def requested_configuration(message, supplied):
    selected = {key: supplied.get(key) for key in
                ("data_version", "rule_version", "score_config_version")}
    patterns = {
        "data_version": ("数据版本", r"数据版本\s*[:：]?\s*([A-Za-z0-9._-]+)"),
        "rule_version": ("规则版本", r"规则版本\s*[:：]?\s*([A-Za-z0-9._-]+)"),
        "score_config_version": ("评分配置", r"评分配置\s*[:：]?\s*([A-Za-z0-9._-]+)"),
    }
    for key, (label, pattern) in patterns.items():
        if label in message:
            match = re.search(pattern, message)
            if not match:
                raise ValueError(f"请明确指定{label}。")
            selected[key] = match.group(1)
    if "版本" in message and not any(label in message for label, _ in patterns.values()) and "默认" not in message:
        raise ValueError("请写明“数据版本”或“规则版本”，或使用默认版本。")
    _, configuration = resolve_configuration(**selected)
    return {"data_version": configuration["raw_data_version"],
            "rule_version": configuration["rule_version"],
            "score_config_version": configuration["score_config_version"]}


def agent_chat(message, task_id=None, supplied_configuration=None):
    message = message.strip()
    if not message:
        return {"reply": "请输入清洗、评估、查看状态或询问已有报告的请求。"}
    task_id = task_id or latest_task()
    if any(word in message for word in ("状态", "进度", "运行到")):
        status = tool_get_status(task_id) if task_id else None
        if not status:
            return {"reply": "目前没有已启动的任务。"}
        return {"reply": f"任务 {task_id}：{status['status']}；{status['progress']}。", "task_id": task_id}
    explanation_prefix = ("请说明", "说明", "请解释", "解释", "请查看", "查看", "请告诉我", "为什么", "如何")
    request_run = (any(word in message for word in ("清洗", "评估"))
                   and any(word in message for word in ("请", "开始", "执行", "运行", "重新"))
                   and ("重新" in message or not message.startswith(explanation_prefix)))
    if request_run:
        try:
            configuration = requested_configuration(message, supplied_configuration or {})
        except FileNotFoundError as exc:
            return {"reply": f"任务未启动：缺少数据文件 {exc}。请按 README 配置课程数据。", "error": "missing_data"}
        except ValueError as exc:
            return {"reply": f"任务未启动：{exc}", "error": "unsupported_configuration"}
        task_id = uuid.uuid4().hex[:12]
        with LOCK:
            TASKS[task_id] = {"task_id": task_id, "status": "queued", "progress": "等待运行",
                              "stage": "queued", "events": [], "engine": ENGINE_LABEL,
                              "configuration": configuration}
            TASK_STORE.put(TASKS[task_id])
        threading.Thread(target=tool_run_pipeline, args=(task_id, configuration), daemon=True).start()
        return {"reply": f"已启动任务 {task_id}，执行引擎：{ENGINE_LABEL}。", "task_id": task_id}
    if not task_id:
        return {"reply": "尚无评估报告。请先请求清洗并评估 MovieLens 数据。"}
    known_questions = ("评分", "分数", "异常", "删除", "隔离", "去重", "修复", "规则", "依据",
                       "数量", "多少", "记录", "时间", "T1", "T2", "划分", "未解决", "局限",
                       "限制", "质量", "报告", "结果", "清洗", "公式")
    if not any(word in message for word in known_questions):
        return {"reply": "暂不支持这类问题。可以询问本次任务的评分、异常、规则、时间边界或未解决问题。",
                "error": "unsupported_intent", "task_id": task_id}
    report = tool_get_report(task_id)
    if report:
        return {"reply": answer_from_report(message, report), "task_id": task_id}
    status = tool_get_status(task_id)
    return {"reply": (f"任务尚未完成：{status['status']}；{status.get('error') or status['progress']}。"
                      if status else "未找到该任务或报告。"), "task_id": task_id}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            self.send_json({"defaults": REGISTRY["defaults"],
                            "data_versions": list(REGISTRY["data_versions"]),
                            "rule_versions": REGISTRY["rule_versions"],
                            "score_config_versions": REGISTRY["score_config_versions"]})
            return
        if path == "/":
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        pieces = path.strip("/").split("/")
        if len(pieces) == 3 and pieces[:2] == ["api", "tasks"]:
            task_id = pieces[2]
            status = tool_get_status(task_id)
            self.send_json(status or {"error": "task not found"}, 200 if status else 404)
            return
        if len(pieces) == 4 and pieces[:2] == ["api", "tasks"] and pieces[3] == "report":
            report = tool_get_report(pieces[2])
            self.send_json(report or {"error": "report not ready"}, 200 if report else 404)
            return
        if len(pieces) == 4 and pieces[:2] == ["api", "tasks"] and pieces[3] == "manifest":
            task_id = pieces[2]
            if not re.fullmatch(r"[0-9a-f]{12}", task_id):
                self.send_json({"error": "invalid task id"}, 400)
                return
            try:
                self.send_json(verify_manifest(DEFAULT_RUNS / task_id))
            except FileNotFoundError:
                self.send_json({"error": "manifest not ready"}, 404)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 409)
            return
        if len(pieces) == 4 and pieces[:2] == ["api", "tasks"] and pieces[3] == "unresolved":
            task_id = pieces[2]
            if not re.fullmatch(r"[0-9a-f]{12}", task_id):
                self.send_json({"error": "invalid task id"}, 400)
                return
            evidence = DEFAULT_RUNS / task_id / "unresolved_conflicts.jsonl"
            if not evidence.is_file():
                self.send_json({"error": "unresolved evidence not ready"}, 404)
                return
            body = evidence.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if len(pieces) == 4 and pieces[:2] == ["api", "tasks"] and pieces[3] == "sample":
            task_id = pieces[2]
            if not re.fullmatch(r"[0-9a-f]{12}", task_id):
                self.send_json({"error": "invalid task id"}, 400)
                return
            query = self.path.partition("?")[2]
            table = query.removeprefix("table=")
            if table not in {"users", "movies", "ratings"}:
                self.send_json({"error": "table must be users, movies or ratings"}, 400)
                return
            sample_path = DEFAULT_RUNS / task_id / f"{table}_clean.dat"
            if not sample_path.is_file():
                self.send_json({"error": "sample not ready"}, 404)
                return
            with sample_path.open(encoding="utf-8") as source:
                sample = [line.rstrip("\n") for _, line in zip(range(10), source)]
            self.send_json({"table": table, "sample": sample})
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self.send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 10000:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            config = {key: body.get(key) for key in
                      ("data_version", "rule_version", "score_config_version")}
            self.send_json(agent_chat(str(body.get("message", "")), body.get("task_id"), config))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)


if __name__ == "__main__":
    if ENGINE not in {"local", "hadoop"}:
        raise SystemExit("LAB2_ENGINE must be local or hadoop")
    host = os.environ.get("LAB2_HOST", "127.0.0.1")
    port = int(os.environ.get("LAB2_PORT", "8765"))
    print(f"Lab2 agent: http://{host}:{port} (engine={ENGINE})", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
