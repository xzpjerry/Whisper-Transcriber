from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from whisper_transcriber.formatters import to_srt, to_vtt


@dataclass(frozen=True)
class JobArtifacts:
    artifact_dir: Path
    audio_path: Path
    transcript_path: Path
    srt_path: Path
    vtt_path: Path
    segments_path: Path


def prepare_job_artifacts(artifacts_root: str | Path, job_id: str) -> JobArtifacts:
    artifact_dir = Path(artifacts_root) / job_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return JobArtifacts(
        artifact_dir=artifact_dir,
        audio_path=artifact_dir / "audio.mp3",
        transcript_path=artifact_dir / "transcript.txt",
        srt_path=artifact_dir / "subtitles.srt",
        vtt_path=artifact_dir / "subtitles.vtt",
        segments_path=artifact_dir / "segments.json",
    )


def write_job_outputs(
    artifacts: JobArtifacts,
    *,
    text: str,
    segments: list[dict[str, Any]],
) -> JobArtifacts:
    artifacts.transcript_path.write_text(text, encoding="utf-8")
    artifacts.srt_path.write_text(to_srt(segments), encoding="utf-8")
    artifacts.vtt_path.write_text(to_vtt(segments), encoding="utf-8")
    artifacts.segments_path.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifacts


def prune_artifacts(
    *,
    artifacts_root: str | Path,
    max_age_hours: int,
    max_count: int,
    now: datetime | None = None,
) -> list[str]:
    reference_now = now or datetime.now(timezone.utc)
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)

    root = Path(artifacts_root)
    if not root.exists():
        return []

    cutoff = reference_now - timedelta(hours=max(0, max_age_hours))
    candidates: list[tuple[Path, datetime]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        candidates.append(
            (
                entry,
                datetime.fromtimestamp(stat.st_mtime, tz=reference_now.tzinfo),
            )
        )

    expired = [item for item in candidates if item[1] < cutoff]
    remaining = [item for item in candidates if item[1] >= cutoff]
    remaining.sort(key=lambda item: (item[1], item[0].name))

    excess = max(0, len(remaining) - max(0, max_count))
    expired.extend(remaining[:excess])
    expired.sort(key=lambda item: (item[1], item[0].name))

    removed: list[str] = []
    for entry, _mtime in expired:
        shutil.rmtree(entry, ignore_errors=True)
        if not entry.exists():
            removed.append(entry.name)
    return removed
