from __future__ import annotations

import sqlite3
from pathlib import Path


# Columns added after the initial schema, applied with ALTER TABLE when missing
ADDED_COLUMNS = (
    ("last_error_detail", "TEXT"),
)


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            youtube_url TEXT NOT NULL,
            display_title TEXT,
            language_hint TEXT,
            status TEXT NOT NULL,
            progress_stage TEXT NOT NULL,
            status_message TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            last_heartbeat_at TEXT,
            worker_id TEXT,
            last_error_code TEXT,
            last_error_message TEXT,
            artifact_dir TEXT,
            last_error_detail TEXT
        )
        """
    )
    _migrate_nullable_display_title(conn)
    _add_missing_columns(conn)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at
        ON jobs (status, created_at)
        """
    )
    conn.commit()


def _migrate_nullable_display_title(conn: sqlite3.Connection) -> None:
    columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
    display_title_column = next(
        (
            column
            for column in columns
            if _pragma_table_info_value(column, "name", 1) == "display_title"
        ),
        None,
    )
    if display_title_column is None or _pragma_table_info_value(display_title_column, "notnull", 3) == 0:
        return

    savepoint_name = "migrate_nullable_display_title"
    conn.execute(f"SAVEPOINT {savepoint_name}")
    try:
        conn.execute("ALTER TABLE jobs RENAME TO jobs__legacy_display_title_not_null")
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                youtube_url TEXT NOT NULL,
                display_title TEXT,
                language_hint TEXT,
                status TEXT NOT NULL,
                progress_stage TEXT NOT NULL,
                status_message TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                last_heartbeat_at TEXT,
                worker_id TEXT,
                last_error_code TEXT,
                last_error_message TEXT,
                artifact_dir TEXT
            )
            """
        )
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
                created_at,
                started_at,
                finished_at,
                last_heartbeat_at,
                worker_id,
                last_error_code,
                last_error_message,
                artifact_dir
            )
            SELECT
                job_id,
                youtube_url,
                display_title,
                language_hint,
                status,
                progress_stage,
                status_message,
                attempt_count,
                created_at,
                started_at,
                finished_at,
                last_heartbeat_at,
                worker_id,
                last_error_code,
                last_error_message,
                artifact_dir
            FROM jobs__legacy_display_title_not_null
            """
        )
        conn.execute("DROP TABLE jobs__legacy_display_title_not_null")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        raise

    conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    existing = {
        _pragma_table_info_value(column, "name", 1)
        for column in conn.execute("PRAGMA table_info(jobs)").fetchall()
    }
    for name, column_type in ADDED_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {column_type}")


def _pragma_table_info_value(
    column: sqlite3.Row | tuple[object, ...],
    key: str,
    index: int,
) -> object:
    if isinstance(column, sqlite3.Row):
        return column[key]
    return column[index]
