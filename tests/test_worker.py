from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised in red phase first
        pytest.fail(f"unable to import {module_name}: {exc}")


def test_process_one_job_completes_claimed_job_and_emits_heartbeats(tmp_path):
    job_store_module = _load_module("whisper_transcriber.job_store")
    worker_module = _load_module("whisper_transcriber.worker")

    store = job_store_module.JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/example",
        display_title="Example Video",
        language_hint="en",
    )

    class RecordingStore:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.heartbeats: list[tuple[str, str, int, str, str]] = []
            self.completed: list[tuple[str, str, int, str | None, str | None, str]] = []
            self.failed: list[tuple[str, str, int, str, str]] = []

        def claim_next_job(self, *, worker_id: str, heartbeat_timeout_seconds: float):
            return self.wrapped.claim_next_job(
                worker_id=worker_id,
                heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            )

        def heartbeat(
            self,
            job_id: str,
            worker_id: str,
            claim_attempt_count: int,
            stage: str,
            message: str,
        ) -> bool:
            self.heartbeats.append(
                (job_id, worker_id, claim_attempt_count, stage, message)
            )
            return self.wrapped.heartbeat(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                stage=stage,
                message=message,
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
            self.completed.append(
                (
                    job_id,
                    worker_id,
                    claim_attempt_count,
                    artifact_dir,
                    display_title,
                    message,
                )
            )
            return self.wrapped.mark_completed(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                artifact_dir=artifact_dir,
                display_title=display_title,
                message=message,
            )

        def mark_failed(
            self,
            job_id: str,
            worker_id: str,
            claim_attempt_count: int,
            error_code: str,
            message: str,
        ) -> bool:
            self.failed.append(
                (job_id, worker_id, claim_attempt_count, error_code, message)
            )
            return self.wrapped.mark_failed(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                error_code=error_code,
                message=message,
            )

        def requeue_stale_jobs(self, heartbeat_timeout_seconds: float) -> int:
            return self.wrapped.requeue_stale_jobs(heartbeat_timeout_seconds)

        def get_job(self, job_id: str):
            return self.wrapped.get_job(job_id)

    recording_store = RecordingStore(store)
    calls: list[dict[str, object]] = []
    cleanup_calls: list[tuple[Path, int, int]] = []

    def fake_run_job(
        *,
        job_id: str,
        youtube_url: str,
        artifacts_root: str | Path,
        media=None,
        language_hint: str | None = None,
        progress_callback=None,
    ):
        calls.append(
            {
                "job_id": job_id,
                "youtube_url": youtube_url,
                "artifacts_root": Path(artifacts_root),
                "language_hint": language_hint,
                "has_progress_callback": progress_callback is not None,
            }
        )
        assert callable(progress_callback)
        progress_callback("downloading", "Downloading source audio")
        progress_callback("transcribing", "Transcribing audio")
        return SimpleNamespace(
            artifact_dir=Path(artifacts_root) / job_id,
            display_title="Recovered Title",
        )

    def fake_prune_artifacts(
        *,
        artifacts_root: str | Path,
        max_age_hours: int,
        max_count: int,
        now=None,
    ):
        cleanup_calls.append((Path(artifacts_root), max_age_hours, max_count))
        return []

    original_run_job = worker_module.run_job
    original_prune_artifacts = worker_module.prune_artifacts
    worker_module.run_job = fake_run_job
    worker_module.prune_artifacts = fake_prune_artifacts
    try:
        processed = worker_module.process_one_job(
            store=recording_store,
            worker_id="worker-1",
            heartbeat_timeout_seconds=30,
            artifacts_root=tmp_path / "artifacts",
            settings=SimpleNamespace(
                max_artifact_age_hours=168,
                max_artifacts=100,
            ),
        )
    finally:
        worker_module.run_job = original_run_job
        worker_module.prune_artifacts = original_prune_artifacts

    refreshed = store.get_job(created.job_id)

    assert processed is True
    assert calls == [
        {
            "job_id": created.job_id,
            "youtube_url": "https://youtu.be/example",
            "artifacts_root": tmp_path / "artifacts",
            "language_hint": "en",
            "has_progress_callback": True,
        }
    ]
    assert recording_store.heartbeats == [
        (
            created.job_id,
            "worker-1",
            1,
            "downloading",
            "Downloading source audio",
        ),
        (
            created.job_id,
            "worker-1",
            1,
            "transcribing",
            "Transcribing audio",
        ),
    ]
    assert recording_store.completed == [
        (
            created.job_id,
            "worker-1",
            1,
            str(tmp_path / "artifacts" / created.job_id),
            "Recovered Title",
            "Completed",
        )
    ]
    assert cleanup_calls == [(tmp_path / "artifacts", 168, 100)]
    assert recording_store.failed == []
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.progress_stage == "completed"
    assert refreshed.status_message == "Completed"
    assert refreshed.display_title == "Recovered Title"
    assert refreshed.artifact_dir == str(tmp_path / "artifacts" / created.job_id)


def test_process_one_job_marks_pipeline_error_as_failed(tmp_path):
    errors_module = _load_module("whisper_transcriber.errors")
    job_store_module = _load_module("whisper_transcriber.job_store")
    worker_module = _load_module("whisper_transcriber.worker")

    store = job_store_module.JobStore(tmp_path / "jobs.sqlite3")
    created = store.create_job(
        youtube_url="https://youtu.be/failure",
        display_title="Failure Case",
        language_hint=None,
    )

    class RecordingStore:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.failed: list[tuple[str, str, int, str, str]] = []

        def claim_next_job(self, *, worker_id: str, heartbeat_timeout_seconds: float):
            return self.wrapped.claim_next_job(
                worker_id=worker_id,
                heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            )

        def heartbeat(
            self,
            job_id: str,
            worker_id: str,
            claim_attempt_count: int,
            stage: str,
            message: str,
        ) -> bool:
            return self.wrapped.heartbeat(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                stage=stage,
                message=message,
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
            return self.wrapped.mark_completed(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                artifact_dir=artifact_dir,
                display_title=display_title,
                message=message,
            )

        def mark_failed(
            self,
            job_id: str,
            worker_id: str,
            claim_attempt_count: int,
            error_code: str,
            message: str,
        ) -> bool:
            self.failed.append(
                (job_id, worker_id, claim_attempt_count, error_code, message)
            )
            return self.wrapped.mark_failed(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=claim_attempt_count,
                error_code=error_code,
                message=message,
            )

        def requeue_stale_jobs(self, heartbeat_timeout_seconds: float) -> int:
            return self.wrapped.requeue_stale_jobs(heartbeat_timeout_seconds)

        def get_job(self, job_id: str):
            return self.wrapped.get_job(job_id)

    recording_store = RecordingStore(store)

    def fake_run_job(**kwargs):
        raise errors_module.DownloadError("download exploded")

    original_run_job = worker_module.run_job
    worker_module.run_job = fake_run_job
    try:
        processed = worker_module.process_one_job(
            store=recording_store,
            worker_id="worker-2",
            heartbeat_timeout_seconds=30,
            artifacts_root=tmp_path / "artifacts",
        )
    finally:
        worker_module.run_job = original_run_job

    refreshed = store.get_job(created.job_id)

    assert processed is True
    assert recording_store.failed == [
        (
            created.job_id,
            "worker-2",
            1,
            "download_error",
            "Unable to download audio from the provided URL.",
        )
    ]
    assert refreshed is not None
    assert refreshed.status == "failed"
    assert refreshed.progress_stage == "failed"
    assert refreshed.status_message == "Unable to download audio from the provided URL."
    assert refreshed.last_error_code == "download_error"
    assert refreshed.last_error_message == "Unable to download audio from the provided URL."


def test_process_one_job_returns_false_when_no_job_is_available(tmp_path):
    job_store_module = _load_module("whisper_transcriber.job_store")
    worker_module = _load_module("whisper_transcriber.worker")

    store = job_store_module.JobStore(tmp_path / "jobs.sqlite3")

    processed = worker_module.process_one_job(
        store=store,
        worker_id="worker-3",
        heartbeat_timeout_seconds=30,
        artifacts_root=tmp_path / "artifacts",
    )

    assert processed is False


def test_main_requeues_stale_jobs_on_startup_and_sleeps_when_idle(tmp_path):
    job_store_module = _load_module("whisper_transcriber.job_store")
    worker_module = _load_module("whisper_transcriber.worker")

    db_path = tmp_path / "jobs.sqlite3"
    store = job_store_module.JobStore(db_path)
    created = store.create_job(
        youtube_url="https://youtu.be/stale",
        display_title="Stale Job",
        language_hint=None,
    )

    claimed = store.claim_next_job(
        worker_id="worker-old",
        heartbeat_timeout_seconds=30,
    )
    assert claimed is not None

    stale_heartbeat = "2000-01-01T00:00:00+00:00"
    with job_store_module.connect_db(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET last_heartbeat_at = ?, status_message = ?
            WHERE job_id = ?
            """,
            (stale_heartbeat, "Worker disappeared", created.job_id),
        )
        conn.commit()

    class RecordingStore:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.requeue_calls: list[float] = []

        def requeue_stale_jobs(self, heartbeat_timeout_seconds: float) -> int:
            self.requeue_calls.append(heartbeat_timeout_seconds)
            return self.wrapped.requeue_stale_jobs(heartbeat_timeout_seconds)

        def get_job(self, job_id: str):
            return self.wrapped.get_job(job_id)

    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    process_calls: list[tuple[str, float, Path]] = []

    def fake_process_one_job(
        *,
        store,
        worker_id: str,
        heartbeat_timeout_seconds: float,
        artifacts_root: Path,
        media=None,
        settings=None,
    ):
        process_calls.append((worker_id, heartbeat_timeout_seconds, Path(artifacts_root)))
        return False

    recording_store = RecordingStore(store)
    original_process_one_job = worker_module.process_one_job
    worker_module.process_one_job = fake_process_one_job
    try:
        exit_code = worker_module.main(
            settings=SimpleNamespace(
                db_path=db_path,
                artifacts_dir=tmp_path / "artifacts",
                heartbeat_seconds=30,
                worker_poll_seconds=0.25,
            ),
            store=recording_store,
            worker_id="worker-main",
            sleep_fn=fake_sleep,
        )
    finally:
        worker_module.process_one_job = original_process_one_job

    refreshed = recording_store.get_job(created.job_id)

    assert exit_code == 0
    assert process_calls == [
        ("worker-main", 90, tmp_path / "artifacts"),
    ]
    assert recording_store.requeue_calls == [90]
    assert sleep_calls == [0.25]
    assert refreshed is not None
    assert refreshed.status == "queued"
    assert refreshed.progress_stage == "queued"
    assert refreshed.status_message == "Requeued after stale worker heartbeat"
    assert refreshed.worker_id is None
    assert refreshed.started_at is None
    assert refreshed.last_heartbeat_at is None
    assert refreshed.attempt_count == 1
