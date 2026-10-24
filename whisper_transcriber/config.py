from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from whisper_transcriber.environment import get_float, get_int, get_path
from whisper_transcriber.job_store import DEFAULT_MAX_JOB_ATTEMPTS


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    db_path: Path
    max_queue_depth: int
    max_artifact_age_hours: int
    max_artifacts: int
    worker_poll_seconds: float
    heartbeat_seconds: float
    max_job_attempts: int = DEFAULT_MAX_JOB_ATTEMPTS


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if environ is None else environ
    return Settings(
        artifacts_dir=get_path("ARTIFACTS_DIR", "./artifacts", environ),
        db_path=get_path("DB_PATH", "./data/transcriber.sqlite3", environ),
        max_queue_depth=get_int("MAX_QUEUE_DEPTH", 5, environ),
        max_artifact_age_hours=get_int("MAX_ARTIFACT_AGE_HOURS", 168, environ),
        max_artifacts=get_int("MAX_ARTIFACTS", 100, environ),
        worker_poll_seconds=get_float("WORKER_POLL_SECONDS", 2, environ),
        heartbeat_seconds=get_float(
            "WORKER_HEARTBEAT_SECONDS",
            get_float("HEARTBEAT_SECONDS", 5, source),
            environ,
        ),
        max_job_attempts=get_int("MAX_JOB_ATTEMPTS", DEFAULT_MAX_JOB_ATTEMPTS, environ),
    )
