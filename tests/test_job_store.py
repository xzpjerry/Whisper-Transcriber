from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from whisper_transcriber.config import load_settings
from whisper_transcriber.db import connect_db, ensure_schema
from whisper_transcriber.job_store import JobStore


def test_load_settings_uses_expected_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in (
        "ARTIFACTS_DIR",
        "DB_PATH",
        "MAX_QUEUE_DEPTH",
        "MAX_ARTIFACT_AGE_HOURS",
        "MAX_ARTIFACTS",
        "WORKER_POLL_SECONDS",
        "WORKER_HEARTBEAT_SECONDS",
        "HEARTBEAT_SECONDS",
        "MAX_JOB_ATTEMPTS",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings()

    assert settings.max_job_attempts == 3

    assert str(settings.artifacts_dir) == "artifacts"
    assert str(settings.db_path) == "data/transcriber.sqlite3"
    assert settings.max_queue_depth == 5
    assert settings.max_artifact_age_hours == 168
    assert settings.max_artifacts == 100
    assert settings.worker_poll_seconds == 2
    assert settings.heartbeat_seconds == 5


def test_load_settings_prefers_worker_heartbeat_seconds_with_legacy_fallback():
    settings = load_settings(
        {
            "WORKER_HEARTBEAT_SECONDS": "7.5",
            "HEARTBEAT_SECONDS": "11",
        }
    )

    fallback_settings = load_settings({"HEARTBEAT_SECONDS": "9"})

    assert settings.heartbeat_seconds == 7.5
    assert fallback_settings.heartbeat_seconds == 9


def test_load_settings_with_explicit_empty_mapping_uses_defaults(monkeypatch):
    monkeypatch.setenv("DB_PATH", "/tmp/ambient.sqlite3")

    settings = load_settings({})

    assert str(settings.db_path) == "data/transcriber.sqlite3"


def test_ensure_schema_creates_jobs_table(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"

    with connect_db(db_path) as conn:
        ensure_schema(conn)

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }

    assert columns >= {
        "job_id",
        "youtube_url",
        "display_title",
        "language_hint",
        "status",
        "progress_stage",
        "status_message",
        "attempt_count",
        "created_at",
        "started_at",
        "finished_at",
        "last_heartbeat_at",
        "worker_id",
        "last_error_code",
        "last_error_message",
        "artifact_dir",
    }


def test_ensure_schema_supports_plain_sqlite_connection(tmp_path):
    db_path = tmp_path / "plain.sqlite3"

    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        columns = conn.execute("PRAGMA table_info(jobs)").fetchall()

    display_title_column = next(column for column in columns if column[1] == "display_title")

    assert display_title_column[3] == 0


def test_ensure_schema_upgrades_not_null_display_title_schema(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                youtube_url TEXT NOT NULL,
                display_title TEXT NOT NULL,
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
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-job",
                "https://youtu.be/legacy",
                "Legacy Title",
                None,
                "queued",
                "queued",
                "Queued",
                0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
        copied_row = conn.execute(
            "SELECT job_id, display_title FROM jobs WHERE job_id = ?",
            ("legacy-job",),
        ).fetchone()

    display_title_column = next(column for column in columns if column[1] == "display_title")

    assert display_title_column[3] == 0
    assert copied_row == ("legacy-job", "Legacy Title")


def test_ensure_schema_rolls_back_display_title_migration_on_failure(tmp_path):
    db_path = tmp_path / "legacy-rollback.sqlite3"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                youtube_url TEXT NOT NULL,
                display_title TEXT NOT NULL,
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
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-job",
                "https://youtu.be/legacy",
                "Legacy Title",
                None,
                "queued",
                "queued",
                "Queued",
                0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

        class FailingConnection:
            def __init__(self, wrapped: sqlite3.Connection) -> None:
                self._wrapped = wrapped

            def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
                if "INSERT INTO jobs (" in sql:
                    raise sqlite3.OperationalError("simulated copy failure")
                return self._wrapped.execute(sql, parameters)

            def commit(self) -> None:
                self._wrapped.commit()

            def __getattr__(self, name: str) -> object:
                return getattr(self._wrapped, name)

        with pytest.raises(sqlite3.OperationalError, match="simulated copy failure"):
            ensure_schema(FailingConnection(conn))

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
        copied_row = conn.execute(
            "SELECT job_id, display_title FROM jobs WHERE job_id = ?",
            ("legacy-job",),
        ).fetchone()

    display_title_column = next(column for column in columns if column[1] == "display_title")

    assert "jobs__legacy_display_title_not_null" not in tables
    assert display_title_column[3] == 1
    assert copied_row == ("legacy-job", "Legacy Title")


def test_create_job_and_read_it_back(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    created = store.create_job(
        youtube_url="https://youtu.be/example",
        display_title="Example Video",
        language_hint="en",
    )

    fetched = store.get_job(created.job_id)
    recent = store.list_recent_jobs(limit=10)

    assert created.status == "queued"
    assert created.progress_stage == "queued"
    assert created.status_message == "Queued"
    assert created.attempt_count == 0
    assert created.youtube_url == "https://youtu.be/example"
    assert created.display_title == "Example Video"
    assert created.language_hint == "en"
    assert fetched == created
    assert recent == [created]


def test_create_job_allows_missing_display_title_until_completion(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    created = store.create_job(
        youtube_url="https://youtu.be/title-later",
        display_title=None,
        language_hint=None,
    )
    claimed = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )

    assert claimed is not None

    completed = store.mark_completed(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=claimed.attempt_count,
        artifact_dir="artifacts/title-later",
        display_title="Recovered Title",
        message="Done",
    )

    refreshed = store.get_job(created.job_id)

    assert created.display_title is None
    assert completed is True
    assert refreshed is not None
    assert refreshed.display_title == "Recovered Title"
    assert refreshed.status == "completed"


def test_claim_next_job_marks_oldest_queued_job_running_atomically(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    first = store.create_job(
        youtube_url="https://youtu.be/first",
        display_title="First",
        language_hint=None,
    )
    second = store.create_job(
        youtube_url="https://youtu.be/second",
        display_title="Second",
        language_hint="fr",
    )

    claimed = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    claimed_again = store.claim_next_job(
        worker_id="worker-2",
        heartbeat_timeout_seconds=30,
    )
    third_claim = store.claim_next_job(
        worker_id="worker-3",
        heartbeat_timeout_seconds=30,
    )

    assert claimed is not None
    assert claimed.job_id == first.job_id
    assert claimed.status == "running"
    assert claimed.progress_stage == "claimed"
    assert claimed.status_message == "Claimed by worker worker-1"
    assert claimed.worker_id == "worker-1"
    assert claimed.attempt_count == 1
    assert claimed.started_at is not None
    assert claimed.last_heartbeat_at is not None

    assert claimed_again is not None
    assert claimed_again.job_id == second.job_id
    assert claimed_again.worker_id == "worker-2"
    assert third_claim is None


def test_requeue_stale_running_job_returns_it_to_queue(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/stale",
        display_title="Stale",
        language_hint=None,
    )

    claimed = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    assert claimed is not None

    stale_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as conn:
        conn.execute(
            """
            UPDATE jobs
            SET last_heartbeat_at = ?, status_message = ?
            WHERE job_id = ?
            """,
            (stale_time.isoformat(), "Worker disappeared", created.job_id),
        )
        conn.commit()

    requeued = store.requeue_stale_jobs(heartbeat_timeout_seconds=30)
    recovered = store.get_job(created.job_id)
    claimed_again = store.claim_next_job(
        worker_id="worker-2",
        heartbeat_timeout_seconds=30,
    )

    assert requeued == 1
    assert recovered is not None
    assert recovered.status == "queued"
    assert recovered.progress_stage == "queued"
    assert recovered.status_message == "Requeued after stale worker heartbeat"
    assert recovered.worker_id is None
    assert recovered.started_at is None
    assert recovered.last_heartbeat_at is None
    assert recovered.attempt_count == 1
    assert claimed_again is not None
    assert claimed_again.job_id == created.job_id
    assert claimed_again.attempt_count == 2


def test_heartbeat_updates_stage_message_and_timestamp(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/heartbeat",
        display_title="Heartbeat",
        language_hint=None,
    )
    claimed = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    assert claimed is not None

    updated = store.heartbeat(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=claimed.attempt_count,
        stage="transcribing",
        message="50% complete",
    )

    refreshed = store.get_job(created.job_id)

    assert updated is True
    assert refreshed is not None
    assert refreshed.progress_stage == "transcribing"
    assert refreshed.status_message == "50% complete"
    assert refreshed.last_heartbeat_at is not None


def test_mark_completed_records_artifact_dir_and_finish_time(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/completed",
        display_title="Completed",
        language_hint=None,
    )
    claimed = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    assert claimed is not None

    updated = store.mark_completed(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=claimed.attempt_count,
        artifact_dir="artifacts/completed-job",
        display_title="Completed",
        message="Done",
    )

    refreshed = store.get_job(created.job_id)

    assert updated is True
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.progress_stage == "completed"
    assert refreshed.status_message == "Done"
    assert refreshed.artifact_dir == "artifacts/completed-job"
    assert refreshed.display_title == "Completed"
    assert refreshed.finished_at is not None


def test_mark_failed_records_error_details_and_finish_time(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/failed",
        display_title="Failed",
        language_hint=None,
    )
    claimed = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    assert claimed is not None

    updated = store.mark_failed(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=claimed.attempt_count,
        error_code="download_error",
        message="Could not fetch media",
    )

    refreshed = store.get_job(created.job_id)

    assert updated is True
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.progress_stage == "failed"
    assert refreshed.status_message == "Could not fetch media"
    assert refreshed.last_error_code == "download_error"
    assert refreshed.last_error_message == "Could not fetch media"
    assert refreshed.finished_at is not None


def test_stale_worker_cannot_mutate_job_after_requeue_and_reclaim(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/reclaimed",
        display_title="Reclaimed",
        language_hint=None,
    )

    claimed_by_a = store.claim_next_job(
        worker_id="worker-a",
        heartbeat_timeout_seconds=30,
    )
    assert claimed_by_a is not None

    stale_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as conn:
        conn.execute(
            "UPDATE jobs SET last_heartbeat_at = ? WHERE job_id = ?",
            (stale_time.isoformat(), created.job_id),
        )
        conn.commit()

    assert store.requeue_stale_jobs(heartbeat_timeout_seconds=30) == 1

    claimed_by_b = store.claim_next_job(
        worker_id="worker-b",
        heartbeat_timeout_seconds=30,
    )
    assert claimed_by_b is not None
    assert claimed_by_b.job_id == created.job_id

    completed = store.mark_completed(
        created.job_id,
        worker_id="worker-a",
        claim_attempt_count=claimed_by_a.attempt_count,
        artifact_dir="artifacts/reclaimed",
        message="stale worker completion",
    )
    heartbeated = store.heartbeat(
        created.job_id,
        worker_id="worker-a",
        claim_attempt_count=claimed_by_a.attempt_count,
        stage="transcribing",
        message="stale heartbeat",
    )
    failed = store.mark_failed(
        created.job_id,
        worker_id="worker-a",
        claim_attempt_count=claimed_by_a.attempt_count,
        error_code="stale",
        message="stale failure",
    )

    refreshed = store.get_job(created.job_id)

    assert completed is False
    assert heartbeated is False
    assert failed is False
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.progress_stage == "claimed"
    assert refreshed.status_message == "Claimed by worker worker-b"
    assert refreshed.worker_id == "worker-b"


def test_same_worker_id_cannot_mutate_job_with_stale_claim_attempt(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/same-worker",
        display_title="Same Worker",
        language_hint=None,
    )

    first_claim = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    assert first_claim is not None

    stale_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as conn:
        conn.execute(
            "UPDATE jobs SET last_heartbeat_at = ? WHERE job_id = ?",
            (stale_time.isoformat(), created.job_id),
        )
        conn.commit()

    assert store.requeue_stale_jobs(heartbeat_timeout_seconds=30) == 1

    second_claim = store.claim_next_job(
        worker_id="worker-1",
        heartbeat_timeout_seconds=30,
    )
    assert second_claim is not None
    assert second_claim.job_id == created.job_id
    assert second_claim.attempt_count == first_claim.attempt_count + 1

    stale_completion = store.mark_completed(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=first_claim.attempt_count,
        artifact_dir="artifacts/same-worker",
        message="stale same worker completion",
    )

    refreshed = store.get_job(created.job_id)

    assert stale_completion is False
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.progress_stage == "claimed"
    assert refreshed.status_message == "Claimed by worker worker-1"
    assert refreshed.worker_id == "worker-1"
    assert refreshed.attempt_count == second_claim.attempt_count


def test_two_claimers_contend_for_one_job_on_same_sqlite_file(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    setup_store = JobStore(db_path)
    queued = setup_store.create_job(
        youtube_url="https://youtu.be/contended",
        display_title=None,
        language_hint=None,
    )

    barrier = threading.Barrier(3)
    claims: list[tuple[str, str | None]] = []
    errors: list[BaseException] = []

    def contender(worker_id: str) -> None:
        store = JobStore(db_path)
        try:
            barrier.wait()
            claimed = store.claim_next_job(
                worker_id=worker_id,
                heartbeat_timeout_seconds=30,
            )
            claims.append((worker_id, None if claimed is None else claimed.job_id))
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    first = threading.Thread(target=contender, args=("worker-a",))
    second = threading.Thread(target=contender, args=("worker-b",))

    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    refreshed = setup_store.get_job(queued.job_id)
    winners = [worker_id for worker_id, job_id in claims if job_id == queued.job_id]
    losers = [worker_id for worker_id, job_id in claims if job_id is None]

    assert errors == []
    assert len(claims) == 2
    assert len(winners) == 1
    assert len(losers) == 1
    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.attempt_count == 1
    assert refreshed.worker_id == winners[0]


def _make_stale(db_path, job_id: str) -> None:
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET last_heartbeat_at = ? WHERE job_id = ?",
            (stale_time.isoformat(), job_id),
        )
        conn.commit()


def test_touch_refreshes_lease_only_for_the_owning_claim(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(youtube_url="https://youtu.be/touch")
    claimed = store.claim_next_job(worker_id="worker-1", heartbeat_timeout_seconds=30)
    assert claimed is not None
    _make_stale(tmp_path / "jobs.sqlite3", created.job_id)

    assert store.touch(created.job_id, worker_id="worker-2", claim_attempt_count=1) is False
    assert store.touch(created.job_id, worker_id="worker-1", claim_attempt_count=1) is True

    refreshed = store.get_job(created.job_id)
    assert refreshed is not None
    assert refreshed.progress_stage == "claimed"
    assert store.requeue_stale_jobs(heartbeat_timeout_seconds=30) == 0


def test_stale_job_that_used_every_attempt_fails_instead_of_requeueing(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    store = JobStore(db_path)
    created = store.create_job(youtube_url="https://youtu.be/crashy")

    for attempt in (1, 2):
        claimed = store.claim_next_job(
            worker_id=f"worker-{attempt}",
            heartbeat_timeout_seconds=30,
            max_attempts=2,
        )
        assert claimed is not None
        assert claimed.attempt_count == attempt
        _make_stale(db_path, created.job_id)

    assert store.requeue_stale_jobs(heartbeat_timeout_seconds=30, max_attempts=2) == 0
    failed = store.get_job(created.job_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.last_error_code == "worker_lost"
    assert failed.finished_at is not None
    assert failed.worker_id is None
    assert store.claim_next_job(worker_id="worker-3", heartbeat_timeout_seconds=30) is None


def test_set_display_title_and_failure_detail_are_persisted(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(youtube_url="https://youtu.be/titled")
    store.claim_next_job(worker_id="worker-1", heartbeat_timeout_seconds=30)

    assert store.set_display_title(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=1,
        display_title="Early Title",
    )
    assert store.mark_failed(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=1,
        error_code="download_error",
        message="Unable to download audio from the provided URL.",
        detail="ERROR: [youtube] abc: This video is unavailable",
        artifact_dir=str(tmp_path / "artifacts" / created.job_id),
    )

    failed = store.get_job(created.job_id)
    assert failed is not None
    assert failed.display_title == "Early Title"
    assert failed.last_error_detail == "ERROR: [youtube] abc: This video is unavailable"
    assert failed.artifact_dir == str(tmp_path / "artifacts" / created.job_id)


def test_ensure_schema_adds_last_error_detail_to_existing_database(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(db_path) as conn:
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
            INSERT INTO jobs (job_id, youtube_url, status, progress_stage, status_message, created_at)
            VALUES ('old-job', 'https://youtu.be/old', 'completed', 'completed', 'Completed', '2026-03-01T00:00:00+00:00')
            """
        )
        conn.commit()

    store = JobStore(db_path)

    old_job = store.get_job("old-job")
    assert old_job is not None
    assert old_job.last_error_detail is None
    assert old_job.force_whisper is False
    assert old_job.transcript_source is None


def test_force_whisper_and_transcript_source_are_persisted(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(youtube_url="https://youtu.be/forced", force_whisper=True)
    claimed = store.claim_next_job(worker_id="worker-1", heartbeat_timeout_seconds=30)
    assert claimed is not None and claimed.force_whisper is True

    store.mark_completed(
        created.job_id,
        worker_id="worker-1",
        claim_attempt_count=claimed.attempt_count,
        artifact_dir="artifacts/x",
        transcript_source="whisper",
        transcript_language="en",
    )

    finished = store.get_job(created.job_id)
    assert finished is not None
    assert finished.transcript_source == "whisper"
    assert finished.transcript_language == "en"
