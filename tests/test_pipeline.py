from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised in red phase first
        pytest.fail(f"unable to import {module_name}: {exc}")


def test_run_job_writes_expected_artifacts_under_a_per_job_directory(tmp_path):
    errors_module = _load_module("whisper_transcriber.errors")
    media_module = _load_module("whisper_transcriber.media")
    pipeline_module = _load_module("whisper_transcriber.pipeline")

    segments = [
        {"start": 0.0, "end": 1.25, "text": "Hello"},
        {"start": 1.5, "end": 2.5, "text": "world"},
    ]
    progress_events: list[tuple[str, str]] = []

    class FakeMediaPipeline:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def download_audio(self, youtube_url: str, working_dir: Path, video=None):
            self.calls.append(("download_audio", youtube_url, working_dir))
            source_path = Path(working_dir) / "episode-source.webm"
            source_path.write_bytes(b"source-audio")
            return media_module.DownloadedMedia(
                source_path=source_path,
                title="Episode 01 / Intro",
            )

        def convert_to_mp3(self, source_audio_path: Path, destination_path: Path):
            self.calls.append(
                (
                    "convert_to_mp3",
                    Path(source_audio_path),
                    Path(destination_path),
                )
            )
            Path(destination_path).write_bytes(b"converted-mp3")
            return Path(destination_path)

        def transcribe(self, mp3_path: Path, language_hint: str | None = None):
            self.calls.append(("transcribe", Path(mp3_path), language_hint))
            return media_module.TranscriptionResult(
                text="Hello world",
                language="en",
                segments=segments,
            )

    media_pipeline = FakeMediaPipeline()

    result = pipeline_module.run_job(
        job_id="job-123",
        youtube_url="https://youtu.be/example",
        artifacts_root=tmp_path / "artifacts",
        language_hint="en",
        media=media_pipeline,
        progress_callback=lambda stage, message: progress_events.append((stage, message)),
    )

    assert not isinstance(result, errors_module.PipelineError)

    expected_job_dir = tmp_path / "artifacts" / "job-123"
    expected_audio = expected_job_dir / "audio.mp3"
    expected_transcript = expected_job_dir / "transcript.txt"
    expected_srt = expected_job_dir / "subtitles.srt"
    expected_vtt = expected_job_dir / "subtitles.vtt"
    expected_segments = expected_job_dir / "segments.json"

    assert result.job_id == "job-123"
    assert result.display_title == "Episode 01 / Intro"
    assert result.language == "en"
    assert result.text == "Hello world"
    assert result.text_len == 11
    assert result.preview == "Hello world"
    assert result.artifact_dir == expected_job_dir
    assert result.artifacts.audio_path == expected_audio
    assert result.artifacts.transcript_path == expected_transcript
    assert result.artifacts.srt_path == expected_srt
    assert result.artifacts.vtt_path == expected_vtt
    assert result.artifacts.segments_path == expected_segments

    assert expected_audio.read_bytes() == b"converted-mp3"
    assert expected_transcript.read_text(encoding="utf-8") == "Hello world"
    assert (
        expected_srt.read_text(encoding="utf-8")
        == "1\n00:00:00,000 --> 00:00:01,250\nHello\n\n"
        "2\n00:00:01,500 --> 00:00:02,500\nworld\n"
    )
    assert (
        expected_vtt.read_text(encoding="utf-8")
        == "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello\n\n"
        "00:00:01.500 --> 00:00:02.500\nworld\n"
    )
    assert json.loads(expected_segments.read_text(encoding="utf-8")) == segments

    assert [stage for stage, _ in progress_events] == [
        "downloading",
        "converting",
        "transcribing",
        "writing",
    ]
    assert media_pipeline.calls == [
        (
            "download_audio",
            "https://youtu.be/example",
            media_pipeline.calls[0][2],
        ),
        (
            "convert_to_mp3",
            media_pipeline.calls[0][2] / "episode-source.webm",
            expected_audio,
        ),
        ("transcribe", expected_audio, "en"),
    ]


def test_run_job_classifies_download_failures_as_typed_pipeline_errors(tmp_path):
    errors_module = _load_module("whisper_transcriber.errors")
    pipeline_module = _load_module("whisper_transcriber.pipeline")

    class FailingMediaPipeline:
        def download_audio(self, youtube_url: str, working_dir: Path, video=None):
            raise RuntimeError("download exploded")

    with pytest.raises(errors_module.DownloadError, match="download exploded") as exc_info:
        pipeline_module.run_job(
            job_id="job-456",
            youtube_url="https://youtu.be/example",
            artifacts_root=tmp_path / "artifacts",
            media=FailingMediaPipeline(),
        )

    assert exc_info.value.error_code == "download_error"
    assert exc_info.value.code == "download_error"
    assert exc_info.value.stage == "downloading"
    assert (
        exc_info.value.user_message
        == "Unable to download audio from the provided URL."
    )
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_run_job_classifies_artifact_setup_failures_as_persistence_errors(tmp_path):
    errors_module = _load_module("whisper_transcriber.errors")
    media_module = _load_module("whisper_transcriber.media")
    pipeline_module = _load_module("whisper_transcriber.pipeline")

    class FakeMediaPipeline:
        def download_audio(self, youtube_url: str, working_dir: Path, video=None):
            source_path = Path(working_dir) / "episode-source.webm"
            source_path.write_bytes(b"source-audio")
            return media_module.DownloadedMedia(
                source_path=source_path,
                title="Episode 01 / Intro",
            )

    original_prepare_job_artifacts = pipeline_module.prepare_job_artifacts

    def fail_prepare_job_artifacts(*args, **kwargs):
        raise OSError("artifact root unavailable")

    pipeline_module.prepare_job_artifacts = fail_prepare_job_artifacts
    try:
        with pytest.raises(
            errors_module.PersistenceError,
            match="artifact root unavailable",
        ) as exc_info:
            pipeline_module.run_job(
                job_id="job-789",
                youtube_url="https://youtu.be/example",
                artifacts_root=tmp_path / "artifacts",
                media=FakeMediaPipeline(),
            )
    finally:
        pipeline_module.prepare_job_artifacts = original_prepare_job_artifacts

    assert exc_info.value.error_code == "persistence_error"
    assert exc_info.value.stage == "writing"
    assert (
        exc_info.value.user_message
        == "Unable to write transcription artifacts."
    )
    assert isinstance(exc_info.value.__cause__, OSError)


def test_run_job_classifies_progress_callback_failures_as_persistence_errors(tmp_path):
    errors_module = _load_module("whisper_transcriber.errors")
    media_module = _load_module("whisper_transcriber.media")
    pipeline_module = _load_module("whisper_transcriber.pipeline")

    class FakeMediaPipeline:
        def __init__(self) -> None:
            self.download_called = False

        def download_audio(self, youtube_url: str, working_dir: Path, video=None):
            self.download_called = True
            source_path = Path(working_dir) / "episode-source.webm"
            source_path.write_bytes(b"source-audio")
            return media_module.DownloadedMedia(
                source_path=source_path,
                title="Episode 01 / Intro",
            )

    media_pipeline = FakeMediaPipeline()

    def fail_progress(stage: str, message: str) -> None:
        raise OSError("heartbeat unavailable")

    with pytest.raises(
        errors_module.PersistenceError,
        match="heartbeat unavailable",
    ) as exc_info:
        pipeline_module.run_job(
            job_id="job-999",
            youtube_url="https://youtu.be/example",
            artifacts_root=tmp_path / "artifacts",
            media=media_pipeline,
            progress_callback=fail_progress,
        )

    assert media_pipeline.download_called is False
    assert exc_info.value.error_code == "persistence_error"
    assert exc_info.value.stage == "downloading"
    assert exc_info.value.user_message == "Unable to persist job progress."
    assert isinstance(exc_info.value.__cause__, OSError)


def test_run_job_reports_title_as_soon_as_download_finishes(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    pipeline_module = _load_module("whisper_transcriber.pipeline")
    events: list[tuple[str, str]] = []

    class FakeMediaPipeline:
        def download_audio(self, youtube_url: str, working_dir: Path, video=None):
            source_path = Path(working_dir) / "source.webm"
            source_path.write_bytes(b"source-audio")
            return media_module.DownloadedMedia(source_path=source_path, title="Episode 02")

        def convert_to_mp3(self, source_audio_path: Path, destination_path: Path):
            Path(destination_path).write_bytes(b"mp3")
            return Path(destination_path)

        def transcribe(self, mp3_path: Path, language_hint: str | None = None):
            return media_module.TranscriptionResult(text="hi", language="en", segments=[])

    pipeline_module.run_job(
        job_id="job-title",
        youtube_url="https://youtu.be/example",
        artifacts_root=tmp_path / "artifacts",
        media=FakeMediaPipeline(),
        progress_callback=lambda stage, message: events.append(("progress", stage)),
        title_callback=lambda title: events.append(("title", title)),
    )

    assert events[:3] == [
        ("progress", "downloading"),
        ("title", "Episode 02"),
        ("progress", "converting"),
    ]


class _SubtitleAwareMedia:
    """Fake media pipeline for the subtitle-first flow."""

    def __init__(
        self,
        media_module,
        *,
        language: str | None,
        subtitle_languages: tuple[str, ...],
        detected: str | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.media_module = media_module
        self.video = media_module.VideoInfo(
            title="Probed Title",
            language=language,
            subtitle_languages=subtitle_languages,
            raw={},
        )
        self.detected = detected
        self.fetch_error = fetch_error
        self.calls: list[str] = []

    def probe(self, youtube_url):
        self.calls.append("probe")
        return self.video

    def fetch_subtitles(self, video, language):
        self.calls.append(f"fetch_subtitles {language}")
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.media_module.TranscriptionResult(
            text="from subtitles",
            language=language,
            segments=[{"start": 0.0, "end": 1.0, "text": "from subtitles"}],
        )

    def download_audio(self, youtube_url, working_dir, video=None):
        self.calls.append(f"download_audio reused={video is self.video}")
        source_path = Path(working_dir) / "source.webm"
        source_path.write_bytes(b"source-audio")
        return self.media_module.DownloadedMedia(source_path=source_path, title="Downloaded Title")

    def detect_language(self, source_audio_path):
        self.calls.append("detect_language")
        return self.detected

    def convert_to_mp3(self, source_audio_path, destination_path):
        self.calls.append("convert_to_mp3")
        Path(destination_path).write_bytes(b"mp3")
        return Path(destination_path)

    def transcribe(self, mp3_path, language_hint=None):
        self.calls.append("transcribe")
        return self.media_module.TranscriptionResult(
            text="from whisper",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "from whisper"}],
        )


def _run_subtitle_job(tmp_path, media, *, language_hint=None, prefer_subtitles=True):
    pipeline_module = _load_module("whisper_transcriber.pipeline")
    titles: list[str] = []
    result = pipeline_module.run_job(
        job_id="job-subs",
        youtube_url="https://youtu.be/example",
        artifacts_root=tmp_path / "artifacts",
        media=media,
        language_hint=language_hint,
        title_callback=titles.append,
        prefer_subtitles=prefer_subtitles,
    )
    return result, titles


def test_run_job_uses_uploaded_subtitles_in_the_spoken_language(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    media = _SubtitleAwareMedia(media_module, language="en", subtitle_languages=("fr", "en"))

    result, titles = _run_subtitle_job(tmp_path, media)

    assert media.calls == ["probe", "fetch_subtitles en"]
    assert titles == ["Probed Title"]
    assert result.source == "subtitles"
    assert result.language == "en"
    assert result.display_title == "Probed Title"
    assert (result.artifact_dir / "transcript.txt").read_text(encoding="utf-8") == "from subtitles"
    assert (result.artifact_dir / "subtitles.srt").exists()
    assert not (result.artifact_dir / "audio.mp3").exists()


def test_run_job_ignores_translated_subtitles_and_reuses_the_probe(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    media = _SubtitleAwareMedia(media_module, language="en", subtitle_languages=("fr", "ja"))

    result, _ = _run_subtitle_job(tmp_path, media)

    assert media.calls == ["probe", "download_audio reused=True", "convert_to_mp3", "transcribe"]
    assert result.source == "whisper"
    assert result.display_title == "Probed Title"


def test_run_job_detects_the_language_before_trusting_subtitles(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    media = _SubtitleAwareMedia(
        media_module,
        language=None,
        subtitle_languages=("zh-Hans",),
        detected="zh",
    )

    result, _ = _run_subtitle_job(tmp_path, media)

    assert media.calls == [
        "probe",
        "download_audio reused=True",
        "detect_language",
        "fetch_subtitles zh-Hans",
    ]
    assert result.source == "subtitles"


def test_run_job_does_not_detect_when_detection_cannot_pick_a_track(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    translated = _SubtitleAwareMedia(
        media_module, language=None, subtitle_languages=("zh-Hans",), detected="en"
    )
    no_subtitles = _SubtitleAwareMedia(media_module, language=None, subtitle_languages=())

    _run_subtitle_job(tmp_path / "translated", translated)
    _run_subtitle_job(tmp_path / "none", no_subtitles)

    assert translated.calls[-2:] == ["convert_to_mp3", "transcribe"]
    assert "fetch_subtitles zh-Hans" not in translated.calls
    assert "detect_language" not in no_subtitles.calls


def test_run_job_falls_back_to_whisper_when_subtitles_fail(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    media = _SubtitleAwareMedia(
        media_module,
        language=None,
        subtitle_languages=("en",),
        fetch_error=RuntimeError("HTTP Error 429"),
    )

    result, _ = _run_subtitle_job(tmp_path, media, language_hint="en")

    assert media.calls == [
        "probe",
        "fetch_subtitles en",
        "download_audio reused=True",
        "convert_to_mp3",
        "transcribe",
    ]
    assert result.source == "whisper"


def test_run_job_skips_subtitles_when_whisper_is_forced(tmp_path):
    media_module = _load_module("whisper_transcriber.media")
    media = _SubtitleAwareMedia(media_module, language="en", subtitle_languages=("en",))

    result, titles = _run_subtitle_job(tmp_path, media, prefer_subtitles=False)

    assert media.calls == ["download_audio reused=False", "convert_to_mp3", "transcribe"]
    assert titles == ["Downloaded Title"]
    assert result.source == "whisper"
