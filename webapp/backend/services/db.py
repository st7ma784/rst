"""
SQLite-backed job and result storage.

Replaces the in-memory dicts so jobs and results survive container restarts.
Uses the standard-library sqlite3 with a threading.Lock for safe concurrent access
(FastAPI background tasks run in a thread pool).
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path("/tmp/siw_workbench.db")
_lock = threading.Lock()


# ── connection helper ─────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _load(s: Optional[str]) -> Any:
    return json.loads(s) if s else None


# ── schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _lock, _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'queued',
                progress     INTEGER NOT NULL DEFAULT 0,
                current_stage TEXT,
                created_at   TEXT,
                started_at   TEXT,
                completed_at TEXT,
                error        TEXT,
                mode         TEXT,
                parameters   TEXT,
                backend      TEXT
            );
            CREATE TABLE IF NOT EXISTS results (
                job_id           TEXT PRIMARY KEY,
                processing_time  REAL,
                stages           TEXT,
                performance      TEXT,
                output_files     TEXT
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                id       INTEGER PRIMARY KEY DEFAULT 1,
                settings TEXT NOT NULL DEFAULT '{}'
            );
        """)


# ── job CRUD ──────────────────────────────────────────────────────────────────

def upsert_job(job: Dict[str, Any]) -> None:
    with _lock, _conn() as c:
        c.execute("""
            INSERT INTO jobs
                (job_id, status, progress, current_stage, created_at,
                 started_at, completed_at, error, mode, parameters, backend)
            VALUES
                (:job_id, :status, :progress, :current_stage, :created_at,
                 :started_at, :completed_at, :error, :mode, :parameters, :backend)
            ON CONFLICT(job_id) DO UPDATE SET
                status        = excluded.status,
                progress      = excluded.progress,
                current_stage = excluded.current_stage,
                started_at    = excluded.started_at,
                completed_at  = excluded.completed_at,
                error         = excluded.error
        """, {
            "job_id":        job.get("job_id"),
            "status":        job.get("status", "queued"),
            "progress":      job.get("progress", 0),
            "current_stage": job.get("current_stage"),
            "created_at":    str(job.get("created_at", datetime.now())),
            "started_at":    str(job.get("started_at")) if job.get("started_at") else None,
            "completed_at":  str(job.get("completed_at")) if job.get("completed_at") else None,
            "error":         job.get("error"),
            "mode":          str(job.get("mode", "auto")),
            "parameters":    _json(job.get("parameters", {})),
            "backend":       job.get("backend"),
        })


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["parameters"] = _load(d.get("parameters")) or {}
    return d


def list_jobs() -> list:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    jobs = []
    for row in rows:
        d = dict(row)
        d["parameters"] = _load(d.get("parameters")) or {}
        jobs.append(d)
    return jobs


def delete_job(job_id: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        c.execute("DELETE FROM results WHERE job_id = ?", (job_id,))


# ── result CRUD ───────────────────────────────────────────────────────────────

def upsert_result(job_id: str, processing_time: float,
                  stages: Dict, performance: Dict,
                  output_files: list = None) -> None:
    with _lock, _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO results
                (job_id, processing_time, stages, performance, output_files)
            VALUES (?, ?, ?, ?, ?)
        """, (
            job_id,
            processing_time,
            _json(stages),
            _json(performance),
            _json(output_files or []),
        ))


def get_result(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM results WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    return {
        "job_id":          row["job_id"],
        "status":          "completed",
        "processing_time": row["processing_time"],
        "stages":          _load(row["stages"]) or {},
        "performance_metrics": _load(row["performance"]) or {},
        "output_files":    _load(row["output_files"]) or [],
    }


# ── settings CRUD ─────────────────────────────────────────────────────────────

def get_settings() -> Dict[str, Any]:
    with _lock, _conn() as c:
        row = c.execute("SELECT settings FROM app_settings WHERE id = 1").fetchone()
    return _load(row["settings"]) if row else {}


def save_settings(data: Dict[str, Any]) -> None:
    with _lock, _conn() as c:
        c.execute("""
            INSERT INTO app_settings (id, settings) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET settings = excluded.settings
        """, (_json(data),))


# ── result summary ─────────────────────────────────────────────────────────────

def get_result_with_summary(job_id: str) -> Optional[Dict[str, Any]]:
    r = get_result(job_id)
    if not r:
        return None
    fa = r["stages"].get("fitacf", {})
    return {
        "good_ranges":      fa.get("good_ranges"),
        "nranges":          fa.get("nranges"),
        "backend":          fa.get("backend"),
        "processing_time":  r["processing_time"],
    }
