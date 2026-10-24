from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, TypeVar

from whisper_transcriber.artifacts import JobArtifacts, prepare_job_artifacts, write_job_outputs
from whisper_transcriber.errors import (
    ConversionError,
    DownloadError,
    PersistenceError,
    PipelineError,
    TranscriptionError,
)
from whisper_transcriber.media import MediaPipeline, TranscriptionResult


ProgressCallback = Callable[[str, str], None]
TitleCallback = Callable[[str], None]
T = TypeVar("T")


@dataclass(frozen=True)
class JobRunResult:
    job_id: str
    display_title: str
    language: str | None
    text: str
    text_len: int
    preview: str
    artifact_dir: Path
    artifacts: JobArtifacts


def run_job(
    *,
    job_id: str,
    youtube_url: str,
    artifacts_root: str | Path,
    media: MediaPipeline | None = None,
    language_hint: str | None = None,
    progress_callback: ProgressCallback | None = None,
    title_callback: TitleCallback | None = None,
) -> JobRunResult:
    media_pipeline = media or MediaPipeline()

    with TemporaryDirectory(prefix=f"whisper-job-{job_id}-") as temp_dir:
        working_dir = Path(temp_dir)

        _emit_progress(progress_callback, "downloading", "Downloading source audio")
        downloaded = _run_boundary(
            DownloadError,
            lambda: media_pipeline.download_audio(youtube_url, working_dir),
        )
        if title_callback is not None and downloaded.title:
            _run_boundary(PersistenceError, lambda: title_callback(downloaded.title))

        artifacts = _run_boundary(
            PersistenceError,
            lambda: prepare_job_artifacts(artifacts_root, job_id),
        )

        _emit_progress(progress_callback, "converting", "Converting source audio")
        converted_audio_path = _run_boundary(
            ConversionError,
            lambda: media_pipeline.convert_to_mp3(
                downloaded.source_path,
                artifacts.audio_path,
            ),
        )

        _emit_progress(progress_callback, "transcribing", "Transcribing audio")
        transcription = _run_boundary(
            TranscriptionError,
            lambda: media_pipeline.transcribe(
                converted_audio_path,
                language_hint=language_hint,
            ),
        )

        _emit_progress(progress_callback, "writing", "Writing job artifacts")
        _run_boundary(
            PersistenceError,
            lambda: write_job_outputs(
                artifacts,
                text=transcription.text,
                segments=transcription.segments,
            ),
        )

    return JobRunResult(
        job_id=job_id,
        display_title=downloaded.title,
        language=transcription.language,
        text=transcription.text,
        text_len=len(transcription.text),
        preview=transcription.text[:1000],
        artifact_dir=artifacts.artifact_dir,
        artifacts=artifacts,
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    if progress_callback is not None:
        try:
            progress_callback(stage, message)
        except Exception as exc:
            raise PersistenceError(
                str(exc),
                stage=stage,
                user_message="Unable to persist job progress.",
            ) from exc


def _run_boundary(
    error_type: type[PipelineError],
    operation: Callable[[], T],
) -> T:
    try:
        return operation()
    except PipelineError:
        raise
    except Exception as exc:
        raise error_type(str(exc)) from exc
