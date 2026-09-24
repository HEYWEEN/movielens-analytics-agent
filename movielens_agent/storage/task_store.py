"""Durable task status for the single-process course application."""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class TaskStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)")
        self._recover_interrupted()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def _recover_interrupted(self):
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT id, payload FROM tasks").fetchall()
            for task_id, payload in rows:
                task = json.loads(payload)
                if task.get("status") in {"queued", "running"}:
                    task.update(status="failed", progress="服务重启，任务执行状态中断",
                                error="服务在任务完成前退出；请重新提交", failed_stage=task.get("stage", "unknown"))
                    db.execute("UPDATE tasks SET payload=? WHERE id=?", (json.dumps(task, ensure_ascii=False), task_id))

    def put(self, task):
        with self.lock, self._connect() as db:
            db.execute("INSERT INTO tasks (id, created_at, payload) VALUES (?, ?, ?)",
                       (task["task_id"], datetime.now(timezone.utc).isoformat(), json.dumps(task, ensure_ascii=False)))

    def update(self, task_id, **changes):
        with self.lock, self._connect() as db:
            row = db.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = json.loads(row[0])
            task.update(changes)
            db.execute("UPDATE tasks SET payload=? WHERE id=?", (json.dumps(task, ensure_ascii=False), task_id))
            return task

    def get(self, task_id):
        with self.lock, self._connect() as db:
            row = db.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def latest_id(self):
        with self.lock, self._connect() as db:
            row = db.execute("SELECT id FROM tasks ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
            return row[0] if row else None
