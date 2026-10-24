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
