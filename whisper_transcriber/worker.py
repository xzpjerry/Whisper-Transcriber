from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

from whisper_transcriber.config import Settings, load_settings
from whisper_transcriber.artifacts import prune_artifacts
from whisper_transcriber.errors import PipelineError
from whisper_transcriber.job_store import JobStore
from whisper_transcriber.media import MediaPipeline
from whisper_transcriber.pipeline import run_job


def process_one_job(
    *,
    store: JobStore,
    worker_id: str,
    heartbeat_timeout_seconds: float,
    artifacts_root: str | Path,
    media: MediaPipeline | None = None,
    settings: Settings | None = None,
) -> bool:
    claimed_job = store.claim_next_job(
        worker_id=worker_id,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )
    if claimed_job is None:
        return False

    progress_callback = _make_progress_callback(
        store=store,
        job_id=claimed_job.job_id,
        worker_id=worker_id,
        claim_attempt_count=claimed_job.attempt_count,
    )

    try:
        result = run_job(
            job_id=claimed_job.job_id,
            youtube_url=claimed_job.youtube_url,
            artifacts_root=artifacts_root,
            media=media,
            language_hint=claimed_job.language_hint,
            progress_callback=progress_callback,
        )
    except PipelineError as exc:
        store.mark_failed(
            claimed_job.job_id,
            worker_id=worker_id,
            claim_attempt_count=claimed_job.attempt_count,
            error_code=exc.error_code,
            message=exc.user_message,
        )
        return True

    store.mark_completed(
        claimed_job.job_id,
        worker_id=worker_id,
        claim_attempt_count=claimed_job.attempt_count,
        artifact_dir=str(result.artifact_dir),
        display_title=result.display_title,
    )
    if settings is not None:
        prune_artifacts(
            artifacts_root=artifacts_root,
            max_age_hours=settings.max_artifact_age_hours,
            max_count=settings.max_artifacts,
        )
    return True


def main(
    *,
    settings: Settings | None = None,
    store: JobStore | None = None,
    worker_id: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    media: MediaPipeline | None = None,
) -> int:
    resolved_settings = settings or load_settings()
    resolved_store = store or JobStore(resolved_settings.db_path)
    resolved_worker_id = worker_id or uuid.uuid4().hex
    lease_timeout_seconds = int(resolved_settings.heartbeat_seconds * 3)

    resolved_store.requeue_stale_jobs(lease_timeout_seconds)

    try:
        while True:
            processed = process_one_job(
                store=resolved_store,
                worker_id=resolved_worker_id,
                heartbeat_timeout_seconds=lease_timeout_seconds,
                artifacts_root=resolved_settings.artifacts_dir,
                media=media,
                settings=resolved_settings,
            )
            if processed:
                continue
            sleep_fn(resolved_settings.worker_poll_seconds)
    except KeyboardInterrupt:
        return 0


def _make_progress_callback(
    *,
    store: JobStore,
    job_id: str,
    worker_id: str,
    claim_attempt_count: int,
) -> Callable[[str, str], None]:
    def progress_callback(stage: str, message: str) -> None:
        updated = store.heartbeat(
            job_id,
            worker_id=worker_id,
            claim_attempt_count=claim_attempt_count,
            stage=stage,
            message=message,
        )
        if not updated:
            raise RuntimeError("job heartbeat rejected")

    return progress_callback
