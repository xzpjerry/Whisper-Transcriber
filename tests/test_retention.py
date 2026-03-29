from __future__ import annotations

import os
from datetime import datetime, timezone

from whisper_transcriber.artifacts import prune_artifacts


def test_prune_artifacts_removes_oldest_and_overage_job_directories(tmp_path):
    now = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)

    fresh = tmp_path / "job-fresh"
    recent = tmp_path / "job-recent"
    mid = tmp_path / "job-mid"
    old = tmp_path / "job-old"
    for path in (fresh, recent, mid, old):
        path.mkdir()

    fresh_mtime = now.timestamp() - 60
    recent_mtime = now.timestamp() - 3600
    mid_mtime = now.timestamp() - 3 * 3600
    old_mtime = now.timestamp() - 8 * 3600
    for path, mtime in (
        (fresh, fresh_mtime),
        (recent, recent_mtime),
        (mid, mid_mtime),
        (old, old_mtime),
    ):
        os.utime(path, (mtime, mtime))

    removed = prune_artifacts(
        artifacts_root=tmp_path,
        max_age_hours=2,
        max_count=2,
        now=now,
    )

    assert removed == ["job-old", "job-mid"]
    assert fresh.exists()
    assert recent.exists()
    assert not mid.exists()
    assert not old.exists()


def test_prune_artifacts_ignores_non_directory_entries(tmp_path):
    now = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)

    job_dir = tmp_path / "job-populated"
    job_dir.mkdir()
    os.utime(job_dir, (now.timestamp(), now.timestamp()))
    stray_file = tmp_path / "README.txt"
    stray_file.write_text("keep me", encoding="utf-8")

    removed = prune_artifacts(
        artifacts_root=tmp_path,
        max_age_hours=24,
        max_count=1,
        now=now,
    )

    assert removed == []
    assert job_dir.exists()
    assert stray_file.exists()
