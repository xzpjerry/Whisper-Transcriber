from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from whisper_transcriber.db import connect_db, ensure_schema


# Total claims a job may get before a stale lease marks it failed instead of requeued
DEFAULT_MAX_JOB_ATTEMPTS = 3
WORKER_LOST_ERROR_CODE = "worker_lost"
WORKER_LOST_MESSAGE = "The worker stopped repeatedly while processing this job."

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    youtube_url: str
    display_title: str | None
    language_hint: str | None
    status: str
    progress_stage: str
    status_message: str
    attempt_count: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    last_heartbeat_at: str | None
    worker_id: str | None
    last_error_code: str | None
    last_error_message: str | None
    artifact_dir: str | None
    last_error_detail: str | None = None


class JobStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        with connect_db(self.db_path) as conn:
            ensure_schema(conn)

    def create_job(
        self,
        youtube_url: str,
        display_title: str | None = None,
        language_hint: str | None = None,
    ) -> JobRecord:
        now = utc_now().isoformat()
        job_id = uuid.uuid4().hex
        with connect_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id,
                    youtube_url,
                    display_title,
                    language_hint,
                    status,
                    progress_stage,
                    status_message,
                    attempt_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    youtube_url,
                    display_title,
                    language_hint,
                    "queued",
                    "queued",
                    "Queued",
                    0,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_job(row)

    def list_recent_jobs(self, limit: int = 20) -> list[JobRecord]:
        with connect_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def get_job(self, job_id: str) -> JobRecord | None:
        with connect_db(self.db_path) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _row_to_job(row)

    def claim_next_job(
        self,
        worker_id: str,
        heartbeat_timeout_seconds: int,
        max_attempts: int = DEFAULT_MAX_JOB_ATTEMPTS,
    ) -> JobRecord | None:
        now = utc_now().isoformat()
        with connect_db(self.db_path) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                _requeue_stale_jobs(conn, heartbeat_timeout_seconds, max_attempts)
                row = conn.execute(
                    """
                    SELECT job_id
                    FROM jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    conn.commit()
                    return None

                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?,
                        progress_stage = ?,
                        status_message = ?,
                        attempt_count = attempt_count + 1,
                        started_at = ?,
                        finished_at = NULL,
                        last_heartbeat_at = ?,
                        worker_id = ?,
                        last_error_code = NULL,
                        last_error_message = NULL,
                        last_error_detail = NULL
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (
                        "running",
                        "claimed",
                        f"Claimed by worker {worker_id}",
                        now,
                        now,
                        worker_id,
                        row["job_id"],
                    ),
                )
                claimed = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?",
                    (row["job_id"],),
                ).fetchone()
                conn.commit()
            except sqlite3.Error:
                conn.rollback()
                raise
        if claimed is None:
            return None
        return _row_to_job(claimed)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        claim_attempt_count: int,
        stage: str,
        message: str,
    ) -> bool:
        now = utc_now().isoformat()
        with connect_db(self.db_path) as conn:
            return _update_owned_running_job(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                fields={
                    "progress_stage": stage,
                    "status_message": message,
                    "last_heartbeat_at": now,
                },
            )

    def touch(
        self,
        job_id: str,
        worker_id: str,
        claim_attempt_count: int,
    ) -> bool:
        """Refresh the lease without changing the visible progress."""
        with connect_db(self.db_path) as conn:
            return _update_owned_running_job(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                fields={"last_heartbeat_at": utc_now().isoformat()},
            )

    def set_display_title(
        self,
        job_id: str,
        worker_id: str,
        claim_attempt_count: int,
        display_title: str,
    ) -> bool:
        with connect_db(self.db_path) as conn:
            return _update_owned_running_job(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                fields={"display_title": display_title},
            )

    def mark_completed(
        self,
        job_id: str,
        worker_id: str,
        claim_attempt_count: int,
        artifact_dir: str | None,
        display_title: str | None = None,
        message: str = "Completed",
    ) -> bool:
        now = utc_now().isoformat()
        fields: dict[str, str | None] = {
            "status": "completed",
            "progress_stage": "completed",
            "status_message": message,
            "finished_at": now,
            "last_heartbeat_at": now,
            "artifact_dir": artifact_dir,
        }
        if display_title is not None:
            fields["display_title"] = display_title
        with connect_db(self.db_path) as conn:
            return _update_owned_running_job(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                fields=fields,
            )

    def mark_failed(
        self,
        job_id: str,
        worker_id: str,
        claim_attempt_count: int,
        error_code: str,
        message: str,
        detail: str | None = None,
        artifact_dir: str | None = None,
    ) -> bool:
        now = utc_now().isoformat()
        fields: dict[str, str | None] = {
            "status": "failed",
            "progress_stage": "failed",
            "status_message": message,
            "finished_at": now,
            "last_heartbeat_at": now,
            "last_error_code": error_code,
            "last_error_message": message,
            "last_error_detail": detail,
        }
        if artifact_dir is not None:
            fields["artifact_dir"] = artifact_dir
        with connect_db(self.db_path) as conn:
            return _update_owned_running_job(
                conn,
                job_id=job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                fields=fields,
            )

    def requeue_stale_jobs(
        self,
        heartbeat_timeout_seconds: int,
        max_attempts: int = DEFAULT_MAX_JOB_ATTEMPTS,
    ) -> int:
        with connect_db(self.db_path) as conn:
            requeued = _requeue_stale_jobs(conn, heartbeat_timeout_seconds, max_attempts)
            conn.commit()
        return requeued


def _requeue_stale_jobs(
    conn: sqlite3.Connection,
    heartbeat_timeout_seconds: int,
    max_attempts: int = DEFAULT_MAX_JOB_ATTEMPTS,
) -> int:
    now = utc_now()
    cutoff = (now - timedelta(seconds=heartbeat_timeout_seconds)).isoformat()
    # Jobs that already used every attempt fail instead of looping forever.
    conn.execute(
        """
        UPDATE jobs
        SET status = 'failed',
            progress_stage = 'failed',
            status_message = ?,
            finished_at = ?,
            worker_id = NULL,
            last_error_code = ?,
            last_error_message = ?,
            last_error_detail = ?
        WHERE status = 'running'
          AND COALESCE(last_heartbeat_at, started_at, created_at) < ?
          AND attempt_count >= ?
        """,
        (
            WORKER_LOST_MESSAGE,
            now.isoformat(),
            WORKER_LOST_ERROR_CODE,
            WORKER_LOST_MESSAGE,
            "Worker heartbeat went stale on every attempt.",
            cutoff,
            max_attempts,
        ),
    )
    cursor = conn.execute(
        """
        UPDATE jobs
        SET status = ?,
            progress_stage = ?,
            status_message = ?,
            started_at = NULL,
            finished_at = NULL,
            last_heartbeat_at = NULL,
            worker_id = NULL
        WHERE status = 'running'
          AND COALESCE(last_heartbeat_at, started_at, created_at) < ?
        """,
        (
            "queued",
            "queued",
            "Requeued after stale worker heartbeat",
            cutoff,
        ),
    )
    return cursor.rowcount


def _update_owned_running_job(
    conn: sqlite3.Connection,
    job_id: str,
    worker_id: str,
    claim_attempt_count: int,
    fields: dict[str, str | None],
) -> bool:
    assignments = ", ".join(f"{column} = ?" for column in fields)
    values = list(fields.values()) + [job_id, worker_id, claim_attempt_count]
    cursor = conn.execute(
        f"""
        UPDATE jobs
        SET {assignments}
        WHERE job_id = ?
          AND status = 'running'
          AND worker_id = ?
          AND attempt_count = ?
        """,
        values,
    )
    conn.commit()
    return cursor.rowcount == 1


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        youtube_url=row["youtube_url"],
        display_title=row["display_title"],
        language_hint=row["language_hint"],
        status=row["status"],
        progress_stage=row["progress_stage"],
        status_message=row["status_message"],
        attempt_count=row["attempt_count"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        last_heartbeat_at=row["last_heartbeat_at"],
        worker_id=row["worker_id"],
        last_error_code=row["last_error_code"],
        last_error_message=row["last_error_message"],
        artifact_dir=row["artifact_dir"],
        last_error_detail=row["last_error_detail"],
    )
