from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from whisper_transcriber.config import Settings, load_settings
from whisper_transcriber.artifacts import JOB_LOG_FILENAME, job_artifact_dir, prune_artifacts
from whisper_transcriber.errors import PipelineError
from whisper_transcriber.job_store import DEFAULT_MAX_JOB_ATTEMPTS, JobStore
from whisper_transcriber.media import MediaPipeline
from whisper_transcriber.pipeline import run_job


logger = logging.getLogger(__name__)
# Captures every module in the package, including yt-dlp output routed via media.py
PACKAGE_LOGGER_NAME = "whisper_transcriber"
# Error detail stored on the job row; the full traceback lives in the job log
MAX_ERROR_DETAIL_CHARS = 2000
INTERNAL_ERROR_CODE = "internal_error"
INTERNAL_ERROR_MESSAGE = "Job failed because of an unexpected worker error."


def process_one_job(
    *,
    store: JobStore,
    worker_id: str,
    heartbeat_timeout_seconds: float,
    artifacts_root: str | Path,
    media: MediaPipeline | None = None,
    settings: Settings | None = None,
    max_attempts: int = DEFAULT_MAX_JOB_ATTEMPTS,
    heartbeat_interval_seconds: float | None = None,
) -> bool:
    claimed_job = store.claim_next_job(
        worker_id=worker_id,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        max_attempts=max_attempts,
    )
    if claimed_job is None:
        return False

    job_id = claimed_job.job_id
    attempt = claimed_job.attempt_count
    job_dir = job_artifact_dir(artifacts_root, job_id)
    interval = heartbeat_interval_seconds or heartbeat_timeout_seconds / 3

    with _job_log(job_dir):
        logger.info(
            "Claimed job %s (attempt %d) for %s",
            job_id,
            attempt,
            claimed_job.youtube_url,
        )
        started = time.monotonic()
        try:
            with _LeaseKeeper(
                store=store,
                job_id=job_id,
                worker_id=worker_id,
                claim_attempt_count=attempt,
                interval_seconds=interval,
            ):
                result = run_job(
                    job_id=job_id,
                    youtube_url=claimed_job.youtube_url,
                    artifacts_root=artifacts_root,
                    media=media,
                    language_hint=claimed_job.language_hint,
                    progress_callback=_make_progress_callback(
                        store=store,
                        job_id=job_id,
                        worker_id=worker_id,
                        claim_attempt_count=attempt,
                    ),
                    title_callback=_make_title_callback(
                        store=store,
                        job_id=job_id,
                        worker_id=worker_id,
                        claim_attempt_count=attempt,
                    ),
                )
        except Exception as exc:
            if isinstance(exc, PipelineError):
                error_code, message = exc.error_code, exc.user_message
            else:
                error_code, message = INTERNAL_ERROR_CODE, INTERNAL_ERROR_MESSAGE
            logger.exception("Job %s failed (%s)", job_id, error_code)
            store.mark_failed(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=attempt,
                error_code=error_code,
                message=message,
                detail=_error_detail(exc),
                artifact_dir=str(job_dir) if job_dir.exists() else None,
            )
        else:
            logger.info("Job %s completed in %.1fs", job_id, time.monotonic() - started)
            if not store.mark_completed(
                job_id,
                worker_id=worker_id,
                claim_attempt_count=attempt,
                artifact_dir=str(result.artifact_dir),
                display_title=result.display_title,
            ):
                logger.warning("Job %s finished after its lease was lost", job_id)

    # Failed jobs leave directories too, so prune after every job.
    if settings is not None:
        _prune_artifacts(artifacts_root, settings)
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
    max_attempts = resolved_settings.max_job_attempts

    resolved_store.requeue_stale_jobs(lease_timeout_seconds, max_attempts=max_attempts)
    logger.info("Worker %s started", resolved_worker_id)

    try:
        while True:
            processed = process_one_job(
                store=resolved_store,
                worker_id=resolved_worker_id,
                heartbeat_timeout_seconds=lease_timeout_seconds,
                artifacts_root=resolved_settings.artifacts_dir,
                media=media,
                settings=resolved_settings,
                max_attempts=max_attempts,
                heartbeat_interval_seconds=resolved_settings.heartbeat_seconds,
            )
            if processed:
                continue
            sleep_fn(resolved_settings.worker_poll_seconds)
    except KeyboardInterrupt:
        return 0


def _prune_artifacts(artifacts_root: str | Path, settings: Settings) -> None:
    try:
        prune_artifacts(
            artifacts_root=artifacts_root,
            max_age_hours=settings.max_artifact_age_hours,
            max_count=settings.max_artifacts,
        )
    except OSError:
        logger.exception("Pruning old artifacts failed")


class _LeaseKeeper:
    """Refreshes a running job's heartbeat in the background.

    Progress updates only happen between stages, while a single download or
    transcription can run far longer than the lease timeout.
    """

    def __init__(
        self,
        *,
        store: JobStore,
        job_id: str,
        worker_id: str,
        claim_attempt_count: int,
        interval_seconds: float,
    ) -> None:
        self._store = store
        self._job_id = job_id
        self._worker_id = worker_id
        self._claim_attempt_count = claim_attempt_count
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-{job_id}",
            daemon=True,
        )

    def __enter__(self) -> _LeaseKeeper:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                kept = self._store.touch(
                    self._job_id,
                    worker_id=self._worker_id,
                    claim_attempt_count=self._claim_attempt_count,
                )
            except Exception:
                logger.exception("Heartbeat for job %s failed", self._job_id)
                continue
            if not kept:
                logger.warning("Lost the lease for job %s", self._job_id)
                return


@contextmanager
def _job_log(job_dir: Path) -> Iterator[None]:
    """Mirror package log output into the job's own log file while it runs.

    A log file that cannot be opened must not take the worker down; the job
    still runs and fails through the pipeline if the directory is unusable.
    """
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(job_dir / JOB_LOG_FILENAME, encoding="utf-8")
    except OSError:
        logger.exception("Could not open the job log in %s", job_dir)
        yield
        return
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_level = package_logger.level
    package_logger.setLevel(logging.DEBUG)
    package_logger.addHandler(handler)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(previous_level)
        handler.close()


def _error_detail(exc: BaseException) -> str:
    detail = str(exc).strip() or type(exc).__name__
    if len(detail) > MAX_ERROR_DETAIL_CHARS:
        detail = detail[:MAX_ERROR_DETAIL_CHARS] + "…"
    return detail


def _make_progress_callback(
    *,
    store: JobStore,
    job_id: str,
    worker_id: str,
    claim_attempt_count: int,
) -> Callable[[str, str], None]:
    def progress_callback(stage: str, message: str) -> None:
        logger.info("[%s] %s", stage, message)
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


def _make_title_callback(
    *,
    store: JobStore,
    job_id: str,
    worker_id: str,
    claim_attempt_count: int,
) -> Callable[[str], None]:
    def title_callback(display_title: str) -> None:
        logger.info("Title: %s", display_title)
        if not store.set_display_title(
            job_id,
            worker_id=worker_id,
            claim_attempt_count=claim_attempt_count,
            display_title=display_title,
        ):
            raise RuntimeError("job title update rejected")

    return title_callback
