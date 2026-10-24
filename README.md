# Whisper Transcriber

Internal FastAPI app for transcribing YouTube audio with `mlx-whisper`. The web process accepts jobs and shows status pages. A separate worker process claims queued jobs, runs the pipeline, and writes per-job artifacts.

## Prerequisites

- Python 3.11+
- `ffmpeg` installed and available on `PATH`
- A JavaScript runtime on `PATH` (`deno`, `node`, `bun`, or `quickjs`); yt-dlp needs one to solve YouTube's challenges
- Access to the configured Whisper model path or Hugging Face repo id

Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Install development/test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Running

Start the web app:

```bash
python app.py
```

Start the worker in a second terminal or service:

```bash
python worker.py
```

The app and worker should point at the same SQLite database and artifacts directory.

## Uploaded Subtitles

When a video already has uploaded subtitles in its spoken language, the worker uses them instead of running Whisper, which takes seconds instead of minutes. YouTube's automatic captions are never used; Whisper is more accurate.

The spoken language comes from the job's language hint, then from YouTube's metadata. If neither is known and the video has uploaded subtitles, the worker downloads the audio and lets Whisper detect the language from the first 30 seconds. Subtitle tracks in any other language are translations and are ignored. If fetching the subtitles fails, the job falls back to Whisper.

Jobs answered from subtitles have no `audio.mp3`. The job page shows where the transcript came from. To always use Whisper, tick **Always transcribe with Whisper** on the submit form, or send `"force_whisper": true` to `/api/transcribe`.

## Environment Variables

Defaults are shown in parentheses.

- `ARTIFACTS_DIR` (`./artifacts`): root directory for per-job artifact folders.
- `DB_PATH` (`./data/transcriber.sqlite3`): SQLite database file.
- `MLX_WHISPER_MODEL` (`~/.lmstudio/models/mlx-community/whisper-large-v3-turbo`): model path or Hugging Face repo id.
- `MAX_QUEUE_DEPTH` (`5`): maximum queued + running jobs accepted by the web app.
- `MAX_ARTIFACT_AGE_HOURS` (`168`): artifact retention window in hours.
- `MAX_ARTIFACTS` (`100`): maximum number of job artifact directories to keep.
- `WORKER_POLL_SECONDS` (`2`): idle worker sleep interval.
- `WORKER_HEARTBEAT_SECONDS` (`5`): worker heartbeat interval.
- `HEARTBEAT_SECONDS` (`5`): legacy fallback for worker heartbeat configuration.
- `MAX_JOB_ATTEMPTS` (`3`): how many times a job whose worker died may be claimed before it is marked failed.

A running job's lease expires after 3× the heartbeat interval. The worker refreshes it in the background for the whole run, so long downloads and transcriptions are not mistaken for dead workers.

## Directory Layout

- `data/` stores the SQLite database and other local state.
- `artifacts/<job_id>/` stores one job's output.
- Each job directory may include:
  - `audio.mp3` (not for jobs answered from uploaded subtitles)
  - `transcript.txt`
  - `subtitles.srt`
  - `subtitles.vtt`
  - `segments.json`
  - `job.log`: the worker's log for this job, including yt-dlp warnings and the full traceback on failure

Artifact cleanup runs on web startup and after every job the worker finishes, successful or not. It is bounded by both age and count, so old job folders are pruned automatically.

## Health Check

`GET /healthz` returns the current environment check status. It validates:

- `ffmpeg` availability
- Whisper model resolution
- JavaScript runtime availability for yt-dlp
- artifact directory writability

If it returns `503`, check the `detail` values in the JSON response and fix the failing dependency or path.

## Troubleshooting

- A job failed: open its page, expand **Details** under the error, and check `job.log` in the artifact list.
- Downloads fail with "The page needs to be reloaded" or "No video formats found": YouTube changed again. Update yt-dlp with `python -m pip install -U "yt-dlp[default]" yt-dlp-ejs`.
- `429 Too Many Requests`: the queue is full. Wait for existing jobs to finish or raise `MAX_QUEUE_DEPTH`.
- `503` from `/healthz`: `ffmpeg`, the model path, the JavaScript runtime, or the artifacts directory is misconfigured.
- Model not found: verify `MLX_WHISPER_MODEL` points to a local model directory or a valid Hugging Face repo id.
- Artifact cleanup not happening: confirm both the web app and worker can write to `ARTIFACTS_DIR`.
- Job stays queued: the worker is not running, or it cannot claim jobs from the database.

## Deployment

Run the app and worker behind VPN access and reverse-proxy basic auth. This service is intended for a small trusted internal group, not public internet exposure.

Typical deployment shape:

1. Web app behind a reverse proxy.
2. Basic auth or equivalent access control at the proxy.
3. Worker process on the same host with the same `DB_PATH` and `ARTIFACTS_DIR`.
4. Persistent storage for `data/` and `artifacts/`.

The app exposes durable job pages and shared job history, so do not publish it directly to the internet without an access-control layer.
