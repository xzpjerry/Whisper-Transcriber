from __future__ import annotations

import logging
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
from whisper_transcriber.media import (
    MediaPipeline,
    TranscriptionResult,
    VideoInfo,
    pick_subtitle_language,
)


ProgressCallback = Callable[[str, str], None]
TitleCallback = Callable[[str], None]
T = TypeVar("T")
SOURCE_SUBTITLES = "subtitles"
SOURCE_WHISPER = "whisper"

logger = logging.getLogger(__name__)


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
    source: str = SOURCE_WHISPER


def run_job(
    *,
    job_id: str,
    youtube_url: str,
    artifacts_root: str | Path,
    media: MediaPipeline | None = None,
    language_hint: str | None = None,
    progress_callback: ProgressCallback | None = None,
    title_callback: TitleCallback | None = None,
    prefer_subtitles: bool = False,
) -> JobRunResult:
    """Transcribe a video, using its uploaded subtitles instead of Whisper when they
    are in the spoken language and ``prefer_subtitles`` is set."""
    media_pipeline = media or MediaPipeline()

    with TemporaryDirectory(prefix=f"whisper-job-{job_id}-") as temp_dir:
        working_dir = Path(temp_dir)
        video: VideoInfo | None = None
        spoken_language = language_hint

        if prefer_subtitles:
            _emit_progress(progress_callback, "downloading", "Checking the video's subtitles")
            video = _run_boundary(DownloadError, lambda: media_pipeline.probe(youtube_url))
            _report_title(title_callback, video.title)
            spoken_language = language_hint or video.language
            if spoken_language:
                result = _try_subtitles(
                    media_pipeline,
                    video=video,
                    spoken_language=spoken_language,
                    job_id=job_id,
                    artifacts_root=artifacts_root,
                    progress_callback=progress_callback,
                )
                if result is not None:
                    return result

        _emit_progress(progress_callback, "downloading", "Downloading source audio")
        downloaded = _run_boundary(
            DownloadError,
            lambda: media_pipeline.download_audio(youtube_url, working_dir, video=video),
        )
        if video is None:
            _report_title(title_callback, downloaded.title)
        elif not spoken_language and video.subtitle_languages:
            # Neither the user nor the site says what is spoken, so ask Whisper
            # before trusting subtitles that may be a translation.
            _emit_progress(progress_callback, "detecting", "Detecting the spoken language")
            detected = _detect_language(media_pipeline, downloaded.source_path)
            if detected:
                result = _try_subtitles(
                    media_pipeline,
                    video=video,
                    spoken_language=detected,
                    job_id=job_id,
                    artifacts_root=artifacts_root,
                    progress_callback=progress_callback,
                )
                if result is not None:
                    return result

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

        return _write_result(
            job_id=job_id,
            title=video.title if video is not None and video.title else downloaded.title,
            transcription=transcription,
            artifacts=artifacts,
            source=SOURCE_WHISPER,
            progress_callback=progress_callback,
        )


def _try_subtitles(
    media_pipeline: MediaPipeline,
    *,
    video: VideoInfo,
    spoken_language: str,
    job_id: str,
    artifacts_root: str | Path,
    progress_callback: ProgressCallback | None,
) -> JobRunResult | None:
    language = pick_subtitle_language(video.subtitle_languages, spoken_language)
    if language is None:
        logger.info(
            "No uploaded subtitles in %s (available: %s); using Whisper",
            spoken_language,
            ", ".join(video.subtitle_languages) or "none",
        )
        return None

    _emit_progress(progress_callback, "downloading", f"Downloading {language} subtitles")
    try:
        transcription = media_pipeline.fetch_subtitles(video, language)
    except Exception:
        # Subtitles are a shortcut; any failure falls back to Whisper.
        logger.warning("Fetching %s subtitles failed; using Whisper", language, exc_info=True)
        return None
    logger.info("Using uploaded %s subtitles (%d cues)", language, len(transcription.segments))

    artifacts = _run_boundary(
        PersistenceError,
        lambda: prepare_job_artifacts(artifacts_root, job_id),
    )
    return _write_result(
        job_id=job_id,
        title=video.title,
        transcription=transcription,
        artifacts=artifacts,
        source=SOURCE_SUBTITLES,
        progress_callback=progress_callback,
    )


def _detect_language(media_pipeline: MediaPipeline, source_path: Path) -> str | None:
    try:
        detected = media_pipeline.detect_language(source_path)
    except Exception:
        logger.warning("Language detection failed; using Whisper", exc_info=True)
        return None
    logger.info("Detected spoken language: %s", detected)
    return detected


def _write_result(
    *,
    job_id: str,
    title: str,
    transcription: TranscriptionResult,
    artifacts: JobArtifacts,
    source: str,
    progress_callback: ProgressCallback | None,
) -> JobRunResult:
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
        display_title=title,
        language=transcription.language,
        text=transcription.text,
        text_len=len(transcription.text),
        preview=transcription.text[:1000],
        artifact_dir=artifacts.artifact_dir,
        artifacts=artifacts,
        source=source,
    )


def _report_title(title_callback: TitleCallback | None, title: str) -> None:
    if title_callback is not None and title:
        _run_boundary(PersistenceError, lambda: title_callback(title))


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
