from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_transcriber.config import Settings
from whisper_transcriber.db import connect_db
from whisper_transcriber.job_store import JobStore


def _load_web_module():
    return importlib.import_module("whisper_transcriber.web")


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        artifacts_dir=tmp_path / "artifacts",
        db_path=tmp_path / "jobs.sqlite3",
        max_queue_depth=5,
        max_artifact_age_hours=168,
        max_artifacts=100,
        worker_poll_seconds=2,
        heartbeat_seconds=5,
    )


def _make_client(tmp_path: Path, monkeypatch):
    web_module = _load_web_module()
    settings = _make_settings(tmp_path)
    store = JobStore(settings.db_path)
    monkeypatch.setattr(
        web_module,
        "_run_environment_checks",
        lambda: (
            {
                "ffmpeg": {"ok": True, "detail": "/usr/bin/ffmpeg"},
                "model": {"ok": True, "detail": "/models/whisper"},
                "artifacts": {"ok": True, "detail": str(settings.artifacts_dir)},
            },
            [],
        ),
    )
    app = web_module.create_app(settings=settings, store=store)
    return TestClient(app), store, settings


def _set_job_fields(store: JobStore, job_id: str, **fields):
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [job_id]
    with connect_db(store.db_path) as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", values)
        conn.commit()


def test_form_submission_creates_queued_job_and_redirects_to_detail(tmp_path, monkeypatch):
    client, store, _settings = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/jobs",
        data={
            "youtube_url": "https://www.youtube.com/watch?v=example",
            "language": "en",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs/")

    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = store.get_job(job_id)

    assert job is not None
    assert job.youtube_url == "https://www.youtube.com/watch?v=example"
    assert job.language_hint == "en"
    assert job.status == "queued"
    assert job.progress_stage == "queued"
    assert job.status_message == "Queued"


def test_json_api_enqueues_and_returns_queued_payload(tmp_path, monkeypatch):
    client, store, _settings = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/transcribe",
        json={
            "youtube_url": "https://www.youtube.com/watch?v=example",
            "language": "ja",
        },
    )

    payload = response.json()

    assert response.status_code == 202
    assert payload["status"] == "queued"
    assert payload["progress_stage"] == "queued"
    assert payload["youtube_url"] == "https://www.youtube.com/watch?v=example"
    assert payload["language_hint"] == "ja"
    assert payload["job_id"]
    assert payload["detail_url"] == f"/jobs/{payload['job_id']}"

    job = store.get_job(payload["job_id"])
    assert job is not None
    assert job.status == "queued"


def test_job_detail_page_renders_job_state_and_artifact_links(tmp_path, monkeypatch):
    client, store, settings = _make_client(tmp_path, monkeypatch)
    job = store.create_job(
        youtube_url="https://www.youtube.com/watch?v=detail",
        display_title="Detail Title",
        language_hint="en",
    )
    artifact_dir = settings.artifacts_dir / job.job_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "audio.mp3",
        "transcript.txt",
        "subtitles.srt",
        "subtitles.vtt",
        "segments.json",
    ):
        (artifact_dir / filename).write_text(filename, encoding="utf-8")

    _set_job_fields(
        store,
        job.job_id,
        status="completed",
        progress_stage="completed",
        status_message="Completed",
        finished_at="2026-03-29T00:00:00+00:00",
        artifact_dir=str(artifact_dir),
    )

    response = client.get(f"/jobs/{job.job_id}")

    assert response.status_code == 200
    html = response.text
    assert "Detail Title" in html
    assert "completed" in html
    assert f"/artifacts/{job.job_id}/transcript.txt" in html
    assert f"/artifacts/{job.job_id}/audio.mp3" in html
    assert "data-job-id" in html
    assert "job-status.js" in html


def test_status_api_returns_job_state(tmp_path, monkeypatch):
    client, store, settings = _make_client(tmp_path, monkeypatch)
    job = store.create_job(
        youtube_url="https://www.youtube.com/watch?v=status",
        display_title="Status Title",
        language_hint=None,
    )
    artifact_dir = settings.artifacts_dir / job.job_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _set_job_fields(
        store,
        job.job_id,
        status="running",
        progress_stage="transcribing",
        status_message="Transcribing audio",
        started_at="2026-03-29T00:00:00+00:00",
        last_heartbeat_at="2026-03-29T00:00:05+00:00",
        worker_id="worker-1",
        artifact_dir=str(artifact_dir),
    )

    response = client.get(f"/api/jobs/{job.job_id}")
    payload = response.json()

    assert response.status_code == 200
    assert payload["job_id"] == job.job_id
    assert payload["status"] == "running"
    assert payload["progress_stage"] == "transcribing"
    assert payload["status_message"] == "Transcribing audio"
    assert payload["artifact_dir"] == str(artifact_dir)


def test_jobs_list_page_renders_recent_jobs(tmp_path, monkeypatch):
    client, store, _settings = _make_client(tmp_path, monkeypatch)
    job = store.create_job(
        youtube_url="https://www.youtube.com/watch?v=list",
        display_title="List Title",
        language_hint="en",
    )

    response = client.get("/jobs")

    assert response.status_code == 200
    assert "List Title" in response.text
    assert job.job_id in response.text


def test_healthz_returns_structured_checks_payload(tmp_path, monkeypatch):
    client, _store, _settings = _make_client(tmp_path, monkeypatch)

    response = client.get("/healthz")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["checks"]["ffmpeg"]["ok"] is True
    assert payload["checks"]["model"]["ok"] is True
    assert payload["checks"]["artifacts"]["ok"] is True


def test_web_startup_prunes_artifacts_after_environment_checks(tmp_path, monkeypatch):
    web_module = _load_web_module()
    settings = _make_settings(tmp_path)
    store = JobStore(settings.db_path)
    call_order: list[str] = []

    def fake_checks():
        call_order.append("checks")
        return (
            {
                "ffmpeg": {"ok": True, "detail": "/usr/bin/ffmpeg"},
                "model": {"ok": True, "detail": "/models/whisper"},
                "artifacts": {"ok": True, "detail": str(settings.artifacts_dir)},
            },
            [],
        )

    def fake_prune_artifacts(
        *,
        artifacts_root,
        max_age_hours,
        max_count,
        now=None,
    ):
        call_order.append("cleanup")
        assert artifacts_root == settings.artifacts_dir
        assert max_age_hours == settings.max_artifact_age_hours
        assert max_count == settings.max_artifacts
        return []

    monkeypatch.setattr(web_module, "_run_environment_checks", fake_checks)
    monkeypatch.setattr(web_module, "prune_artifacts", fake_prune_artifacts)

    app = web_module.create_app(settings=settings, store=store)

    with TestClient(app):
        pass

    assert call_order == ["checks", "cleanup"]


def test_queue_limit_rejects_new_submissions_with_friendly_429(tmp_path, monkeypatch):
    client, store, settings = _make_client(tmp_path, monkeypatch)
    for index in range(settings.max_queue_depth):
        store.create_job(
            youtube_url=f"https://www.youtube.com/watch?v=queued-{index}",
            display_title=f"Queued {index}",
            language_hint=None,
        )

    response = client.post(
        "/api/transcribe",
        json={"youtube_url": "https://www.youtube.com/watch?v=overflow"},
    )

    assert response.status_code == 429
    payload = response.json()
    assert payload["detail"] == "The transcription queue is full. Please try again later."
