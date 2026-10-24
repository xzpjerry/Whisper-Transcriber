from __future__ import annotations

import importlib
import json
import logging
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
# Lines of ffmpeg stderr to keep when a conversion fails
FFMPEG_STDERR_TAIL_LINES = 5
# Whisper detects the spoken language from the first 30 seconds, as transcribe() does
LANGUAGE_DETECTION_SECONDS = 30
# Subtitle tracks yt-dlp lists that are not transcripts
NON_TRANSCRIPT_TRACKS = frozenset({"live_chat"})
# Legacy language codes YouTube still uses, mapped to the ISO codes Whisper reports
LANGUAGE_ALIASES = {"iw": "he", "in": "id", "jw": "jv"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedMedia:
    source_path: Path
    title: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[dict[str, Any]]


@dataclass(frozen=True)
class VideoInfo:
    title: str
    # Spoken language from the site's metadata, when it reports one
    language: str | None
    # Languages of uploaded (not automatic) subtitles
    subtitle_languages: tuple[str, ...]
    # Raw yt-dlp info, reused to download without extracting again
    raw: dict[str, Any]


def pick_subtitle_language(
    subtitle_languages: tuple[str, ...] | list[str],
    spoken_language: str,
) -> str | None:
    """Return the uploaded subtitle track in the spoken language, if any.

    Tracks in other languages are translations, not transcripts.
    """
    wanted = spoken_language.strip().lower()
    for language in subtitle_languages:
        if language.lower() == wanted:
            return language
    wanted_primary = _primary_language(wanted)
    for language in subtitle_languages:
        if _primary_language(language) == wanted_primary:
            return language
    return None


def parse_json3_subtitles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        text = "".join(seg.get("utf8", "") for seg in event.get("segs") or [])
        text = " ".join(text.split())
        if not text:
            continue
        start_ms = event.get("tStartMs", 0)
        start = start_ms / 1000
        end = (start_ms + event.get("dDurationMs", 0)) / 1000
        segments.append({"id": len(segments), "start": start, "end": end, "text": text})
    return segments


def _primary_language(language: str) -> str:
    primary = language.lower().replace("_", "-").split("-")[0]
    return LANGUAGE_ALIASES.get(primary, primary)


class MediaPipeline:
    def __init__(self, model_path: str | None = None) -> None:
        configured_model = model_path or DEFAULT_MODEL_PATH
        self.model_path = os.path.expanduser(configured_model)

    def probe(self, youtube_url: str) -> VideoInfo:
        """Read the video's metadata and subtitle list without downloading media."""
        yt_dlp = importlib.import_module("yt_dlp")
        with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        subtitles = info.get("subtitles") or {}
        return VideoInfo(
            title=str(info.get("title") or ""),
            language=info.get("language") or None,
            subtitle_languages=tuple(
                language
                for language, tracks in subtitles.items()
                if language not in NON_TRANSCRIPT_TRACKS and tracks
            ),
            raw=info,
        )

    def fetch_subtitles(self, video: VideoInfo, language: str) -> TranscriptionResult:
        yt_dlp = importlib.import_module("yt_dlp")
        tracks = video.raw["subtitles"][language]
        track = next((item for item in tracks if item.get("ext") == "json3"), None)
        if track is None:
            raise ValueError(f"no json3 track for {language} subtitles")
        with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            payload = json.loads(ydl.urlopen(track["url"]).read())
        segments = parse_json3_subtitles(payload)
        if not segments:
            raise ValueError(f"{language} subtitles are empty")
        return TranscriptionResult(
            text="\n".join(segment["text"] for segment in segments),
            language=language,
            segments=segments,
        )

    def download_audio(
        self,
        youtube_url: str,
        working_dir: str | Path,
        video: VideoInfo | None = None,
    ) -> DownloadedMedia:
        yt_dlp = importlib.import_module("yt_dlp")
        working_dir = Path(working_dir)
        ydl_opts = _ydl_opts(outtmpl=str(working_dir / "%(title)s.%(ext)s"))
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if video is not None:
                # Reuse the probed info instead of extracting the page again.
                info = ydl.process_ie_result(video.raw, download=True)
            else:
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
        try:
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
        except ffmpeg.Error as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            logger.error("ffmpeg stderr:\n%s", stderr)
            tail = "\n".join(stderr.strip().splitlines()[-FFMPEG_STDERR_TAIL_LINES:])
            raise RuntimeError(f"ffmpeg failed: {tail or exc}") from exc
        return destination_path

    def detect_language(self, source_audio_path: str | Path) -> str:
        """Detect the spoken language from the start of the audio with Whisper."""
        ffmpeg = importlib.import_module("ffmpeg")
        mx = importlib.import_module("mlx.core")
        np = importlib.import_module("numpy")
        whisper_audio = importlib.import_module("mlx_whisper.audio")
        whisper_transcribe = importlib.import_module("mlx_whisper.transcribe")

        try:
            raw, _ = ffmpeg.run(
                ffmpeg.input(str(source_audio_path), t=LANGUAGE_DETECTION_SECONDS).output(
                    "pipe:",
                    format="s16le",
                    acodec="pcm_s16le",
                    ac=1,
                    ar=whisper_audio.SAMPLE_RATE,
                ),
                capture_stdout=True,
                capture_stderr=True,
            )
        except ffmpeg.Error as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            tail = "\n".join(stderr.strip().splitlines()[-FFMPEG_STDERR_TAIL_LINES:])
            raise RuntimeError(f"ffmpeg failed: {tail or exc}") from exc
        samples = np.frombuffer(raw, np.int16).flatten().astype(np.float32) / 32768.0

        # Same steps transcribe() uses, so the shared cached model is reused.
        model = whisper_transcribe.ModelHolder.get_model(self.model_path, mx.float16)
        mel = whisper_audio.log_mel_spectrogram(
            samples,
            n_mels=model.dims.n_mels,
            padding=whisper_audio.N_SAMPLES,
        )
        segment = whisper_audio.pad_or_trim(mel, whisper_audio.N_FRAMES, axis=-2)
        _, probs = model.detect_language(segment.astype(mx.float16))
        return max(probs, key=probs.get)

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


def _ydl_opts(**overrides: Any) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "noprogress": True,
        # Route yt-dlp output (including warnings) into the job log.
        "logger": logger,
        "format": "bestaudio/best",
        "cookiesfrombrowser": ("chrome",),
        # YouTube requires a JS runtime to solve its challenges; yt-dlp only
        # enables deno by default, so allow whichever one is installed.
        "js_runtimes": {name: {} for name in JS_RUNTIMES},
    }
    opts.update(overrides)
    return opts
