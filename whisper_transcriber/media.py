from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = os.environ.get(
    "MLX_WHISPER_MODEL",
    os.path.expanduser("~/.lmstudio/models/mlx-community/whisper-large-v3-turbo"),
)
# JS runtimes yt-dlp may use, in order of preference
JS_RUNTIMES = ("deno", "node", "bun", "quickjs")


@dataclass(frozen=True)
class DownloadedMedia:
    source_path: Path
    title: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[dict[str, Any]]


class MediaPipeline:
    def __init__(self, model_path: str | None = None) -> None:
        configured_model = model_path or DEFAULT_MODEL_PATH
        self.model_path = os.path.expanduser(configured_model)

    def download_audio(
        self,
        youtube_url: str,
        working_dir: str | Path,
    ) -> DownloadedMedia:
        yt_dlp = importlib.import_module("yt_dlp")
        working_dir = Path(working_dir)
        ydl_opts = {
            "outtmpl": str(working_dir / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "cookiesfrombrowser": ("chrome",),
            # YouTube requires a JS runtime to solve its challenges; yt-dlp only
            # enables deno by default, so allow whichever one is installed.
            "js_runtimes": {name: {} for name in JS_RUNTIMES},
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
            if not downloaded.exists():
                base = downloaded.with_suffix("")
                for extension in (".m4a", ".webm", ".mp4", ".ogg", ".opus"):
                    candidate = base.with_suffix(extension)
                    if candidate.exists():
                        downloaded = candidate
                        break

        title = str(info.get("title") or downloaded.stem)
        return DownloadedMedia(source_path=downloaded, title=title)

    def convert_to_mp3(
        self,
        source_audio_path: str | Path,
        destination_path: str | Path,
    ) -> Path:
        ffmpeg = importlib.import_module("ffmpeg")
        source_audio_path = Path(source_audio_path)
        destination_path = Path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        stream = ffmpeg.input(str(source_audio_path))
        stream = ffmpeg.output(
            stream,
            str(destination_path),
            ac=1,
            ar=16000,
            audio_bitrate="160k",
            vn=None,
        )
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        return destination_path

    def transcribe(
        self,
        mp3_path: str | Path,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        mlx_whisper = importlib.import_module("mlx_whisper")
        kwargs: dict[str, Any] = {}
        if language_hint:
            kwargs["language"] = language_hint

        result = mlx_whisper.transcribe(
            str(mp3_path),
            path_or_hf_repo=self.model_path,
            **kwargs,
        )
        return TranscriptionResult(
            text=str(result.get("text", "")),
            language=result.get("language"),
            segments=list(result.get("segments") or []),
        )
