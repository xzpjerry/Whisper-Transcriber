from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from whisper_transcriber.artifacts import prune_artifacts
from whisper_transcriber.config import Settings, load_settings
from whisper_transcriber.db import connect_db
from whisper_transcriber.job_store import JobRecord, JobStore


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
DEFAULT_MODEL_PATH = os.environ.get(
    "MLX_WHISPER_MODEL",
    os.path.expanduser("~/.lmstudio/models/mlx-community/whisper-large-v3-turbo"),
)
ARTIFACT_FILENAMES = (
    ("transcript", "Transcript (.txt)", "transcript.txt", True),
    ("srt", "Subtitles (.srt)", "subtitles.srt", False),
    ("vtt", "Subtitles (.vtt)", "subtitles.vtt", False),
    ("segments", "Segments (.json)", "segments.json", False),
    ("audio", "Audio (.mp3)", "audio.mp3", False),
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
_ACTIVE_SETTINGS: Settings | None = None


def _current_settings() -> Settings:
    return _ACTIVE_SETTINGS or load_settings()


def _is_hf_repo_id(value: str) -> bool:
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", value))


def _check_ffmpeg_binary() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "`ffmpeg` executable was not found in PATH. Install ffmpeg and restart the app."
        )
    return ffmpeg_path


def _check_model_location() -> str:
    configured = (DEFAULT_MODEL_PATH or "").strip()
    if not configured:
        raise RuntimeError("`MLX_WHISPER_MODEL` is empty.")

    resolved = os.path.abspath(os.path.expanduser(configured))
    if os.path.exists(resolved):
        return resolved
    if _is_hf_repo_id(configured):
        return configured

    raise RuntimeError(
        f"Model path is invalid: {configured}. Set `MLX_WHISPER_MODEL` to an existing local path "
        "or a Hugging Face repo id (for example: mlx-community/whisper-large-v3-turbo)."
    )


def _check_artifacts_dir() -> str:
    artifacts_dir = _current_settings().artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    probe_path = None
    fd = None
    try:
        fd, probe_path = tempfile.mkstemp(prefix=".write-check-", dir=artifacts_dir)
    except OSError as exc:
        raise RuntimeError(f"`{artifacts_dir}` is not writable: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if probe_path and os.path.exists(probe_path):
            os.unlink(probe_path)
    return str(artifacts_dir)


def _run_environment_checks() -> tuple[dict[str, dict[str, Any]], list[str]]:
    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for key, checker in (
        ("ffmpeg", _check_ffmpeg_binary),
        ("model", _check_model_location),
        ("artifacts", _check_artifacts_dir),
    ):
        try:
            checks[key] = {"ok": True, "detail": checker()}
        except Exception as exc:
            message = str(exc)
            checks[key] = {"ok": False, "detail": message}
            errors.append(f"{key}: {message}")
    return checks, errors


def _job_url(job_id: str) -> str:
    return f"/jobs/{job_id}"


def _api_job_url(job_id: str) -> str:
    return f"/api/jobs/{job_id}"


def _artifact_base(job: JobRecord) -> str | None:
    if not job.artifact_dir:
        return None

    artifact_dir = Path(job.artifact_dir)
    settings = _current_settings()
    root = settings.artifacts_dir.resolve()
    try:
        relative = artifact_dir.resolve().relative_to(root)
    except Exception:
        relative = Path(artifact_dir.name)
    return f"/artifacts/{relative.as_posix()}"


def _artifact_links(job: JobRecord) -> list[dict[str, Any]]:
    base = _artifact_base(job)
    if base is None:
        return []

    links: list[dict[str, Any]] = []
    for kind, label, filename, copyable in ARTIFACT_FILENAMES:
        links.append(
            {
                "kind": kind,
                "label": label,
                "href": f"{base}/{filename}",
                "copyable": copyable,
            }
        )
    return links


def _job_preview(job: JobRecord) -> str | None:
    if not job.artifact_dir:
        return None

    transcript_path = Path(job.artifact_dir) / "transcript.txt"
    if not transcript_path.exists():
        return None
    return transcript_path.read_text(encoding="utf-8")[:1000]


def _job_payload(job: JobRecord) -> dict[str, Any]:
    payload = asdict(job)
    payload["detail_url"] = _job_url(job.job_id)
    payload["api_url"] = _api_job_url(job.job_id)
    payload["artifact_links"] = _artifact_links(job)
    payload["preview"] = _job_preview(job)
    payload["is_terminal"] = job.status in {"completed", "failed"}
    return payload


def _queue_depth(store: JobStore) -> int:
    with connect_db(store.db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM jobs
            WHERE status IN ('queued', 'running')
            """
        ).fetchone()
    return int(row["total"]) if row is not None else 0


def _normalize_url(youtube_url: str) -> str:
    cleaned = (youtube_url or "").strip()
    if not cleaned.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    return cleaned


def _normalize_language(language: str | None) -> str | None:
    cleaned = (language or "").strip()
    return cleaned or None


def _enqueue_job(
    *,
    store: JobStore,
    settings: Settings,
    youtube_url: str,
    language: str | None,
) -> dict[str, Any]:
    if _queue_depth(store) >= settings.max_queue_depth:
        raise HTTPException(
            status_code=429,
            detail="The transcription queue is full. Please try again later.",
        )

    job = store.create_job(
        youtube_url=_normalize_url(youtube_url),
        language_hint=_normalize_language(language),
    )
    return _job_payload(job)


def _job_or_404(store: JobStore, job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def create_app(
    *,
    settings: Settings | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    resolved_settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = resolved_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _, errors = _run_environment_checks()
        if errors:
            bullet_list = "\n".join(f"- {item}" for item in errors)
            raise RuntimeError(f"Startup checks failed:\n{bullet_list}")
        prune_artifacts(
            artifacts_root=resolved_settings.artifacts_dir,
            max_age_hours=resolved_settings.max_artifact_age_hours,
            max_count=resolved_settings.max_artifacts,
        )
        yield

    app = FastAPI(title="Whisper Transcriber", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.store = store or JobStore(resolved_settings.db_path)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount(
        "/artifacts",
        StaticFiles(directory=str(resolved_settings.artifacts_dir)),
        name="artifacts",
    )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        recent_jobs = app.state.store.list_recent_jobs(limit=5)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "title": "Submit a Job",
                "recent_jobs": recent_jobs,
            },
        )

    @app.post("/jobs")
    def create_job(
        youtube_url: str = Form(...),
        language: str = Form(default=""),
    ) -> RedirectResponse:
        job = _enqueue_job(
            store=app.state.store,
            settings=resolved_settings,
            youtube_url=youtube_url,
            language=language,
        )
        return RedirectResponse(job["detail_url"], status_code=303)

    @app.post("/transcribe")
    def create_job_legacy(
        youtube_url: str = Form(...),
        language: str = Form(default=""),
    ) -> RedirectResponse:
        return create_job(youtube_url=youtube_url, language=language)

    @app.get("/jobs")
    def job_list(request: Request) -> HTMLResponse:
        recent_jobs = app.state.store.list_recent_jobs(limit=50)
        return templates.TemplateResponse(
            request,
            "job_list.html",
            {
                "title": "Recent Jobs",
                "jobs": recent_jobs,
            },
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: str) -> HTMLResponse:
        job = _job_or_404(app.state.store, job_id)
        payload = _job_payload(job)
        return templates.TemplateResponse(
            request,
            "job_detail.html",
            {
                "title": f"Job {job.job_id}",
                "job": payload,
                "api_url": payload["api_url"],
                "artifact_links": payload["artifact_links"],
                "preview": payload["preview"],
                "poll_interval_ms": 2000,
            },
        )

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        job = _job_or_404(app.state.store, job_id)
        return JSONResponse(_job_payload(job))

    @app.post("/api/transcribe", status_code=202)
    def enqueue_api(payload: dict[str, Any]) -> JSONResponse:
        job = _enqueue_job(
            store=app.state.store,
            settings=resolved_settings,
            youtube_url=str(payload.get("youtube_url") or ""),
            language=(
                payload.get("language_hint")
                if payload.get("language_hint") is not None
                else payload.get("language")
            ),
        )
        return JSONResponse(job, status_code=202)

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        checks, errors = _run_environment_checks()
        status_code = 200 if not errors else 503
        status = "ok" if not errors else "error"
        return JSONResponse({"status": status, "checks": checks}, status_code=status_code)

    return app
