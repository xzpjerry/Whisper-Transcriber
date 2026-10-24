from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisper_transcriber import media as media_module


def test_download_audio_enables_js_runtimes_and_routes_output_to_logger(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class FakeYoutubeDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, url, download):
            (tmp_path / "Clip.m4a").write_bytes(b"audio")
            return {"title": "Clip"}

        def prepare_filename(self, info):
            return str(tmp_path / "Clip.m4a")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    downloaded = media_module.MediaPipeline(model_path="/models/x").download_audio(
        "https://youtu.be/clip",
        tmp_path,
    )

    assert downloaded.title == "Clip"
    assert set(captured["js_runtimes"]) == set(media_module.JS_RUNTIMES)
    assert captured["logger"] is media_module.logger
    assert "no_warnings" not in captured


def test_convert_to_mp3_surfaces_the_tail_of_ffmpeg_stderr(tmp_path, monkeypatch):
    class FakeFfmpegError(Exception):
        def __init__(self, stderr: bytes):
            super().__init__("ffmpeg error (see stderr output for detail)")
            self.stderr = stderr

    def fake_run(stream, quiet, overwrite_output):
        raise FakeFfmpegError(b"banner\nconfig\nsource.webm: Invalid data found when processing input\n")

    fake_ffmpeg = SimpleNamespace(
        Error=FakeFfmpegError,
        input=lambda path: ("input", path),
        output=lambda stream, path, **kwargs: ("output", stream, path),
        run=fake_run,
    )
    monkeypatch.setitem(sys.modules, "ffmpeg", fake_ffmpeg)

    with pytest.raises(RuntimeError, match="Invalid data found when processing input"):
        media_module.MediaPipeline(model_path="/models/x").convert_to_mp3(
            tmp_path / "source.webm",
            tmp_path / "out" / "audio.mp3",
        )


def test_pick_subtitle_language_only_accepts_the_spoken_language():
    pick = media_module.pick_subtitle_language

    assert pick(("en-GB", "en"), "en") == "en"
    assert pick(("zh-Hant", "zh-Hans"), "zh") == "zh-Hant"
    assert pick(("zh-Hans",), "zh-CN") == "zh-Hans"
    assert pick(("iw",), "he") == "iw"
    # Uploaded tracks in other languages are translations, not transcripts.
    assert pick(("fr", "de", "ja"), "en") is None
    assert pick((), "en") is None


def test_parse_json3_subtitles_skips_empty_cues_and_flattens_lines():
    payload = {
        "events": [
            {"tStartMs": 0, "dDurationMs": 5000},
            {"tStartMs": 67, "dDurationMs": 2533, "segs": [{"utf8": "Hello\nthere"}]},
            {"tStartMs": 2600, "dDurationMs": 100, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 2633, "dDurationMs": 1634, "segs": [{"utf8": "General "}, {"utf8": "Kenobi"}]},
        ]
    }

    assert media_module.parse_json3_subtitles(payload) == [
        {"id": 0, "start": 0.067, "end": 2.6, "text": "Hello there"},
        {"id": 1, "start": 2.633, "end": 4.267, "text": "General Kenobi"},
    ]


def test_probe_lists_uploaded_subtitles_and_download_reuses_the_probed_info(tmp_path, monkeypatch):
    info = {
        "title": "Clip",
        "language": "en",
        "subtitles": {"en": [{"ext": "json3", "url": "u"}], "live_chat": [{"ext": "json"}], "fr": []},
        "automatic_captions": {"de": [{"ext": "json3"}]},
    }
    calls: list[str] = []

    class FakeYoutubeDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def extract_info(self, url, download):
            calls.append(f"extract download={download}")
            return info

        def process_ie_result(self, ie_result, download):
            calls.append(f"process download={download}")
            (tmp_path / "Clip.m4a").write_bytes(b"audio")
            return ie_result

        def prepare_filename(self, info):
            return str(tmp_path / "Clip.m4a")

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    pipeline = media_module.MediaPipeline(model_path="/models/x")

    video = pipeline.probe("https://youtu.be/clip")
    downloaded = pipeline.download_audio("https://youtu.be/clip", tmp_path, video=video)

    assert video.title == "Clip"
    assert video.language == "en"
    assert video.subtitle_languages == ("en",)
    assert downloaded.source_path == tmp_path / "Clip.m4a"
    assert calls == ["extract download=False", "process download=True"]
