# app.py
import os
import json
import shutil
import tempfile
import re
import unicodedata
from datetime import timedelta
from typing import List, Dict

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

import yt_dlp
import ffmpeg
import mlx_whisper

MODEL_PATH = os.environ.get(
    "MLX_WHISPER_MODEL",
    os.path.expanduser("~/.lmstudio/models/mlx-community/whisper-large-v3-turbo"),
)

# JS runtimes yt-dlp may use, in order of preference
JS_RUNTIMES = ("deno", "node", "bun", "quickjs")

app = FastAPI(title="YouTube → MP3 → Whisper (MLX)")

# simple static dir for artifacts
ARTIFACTS_DIR = os.path.abspath("./artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")

INDEX_HTML = Template("""
<!doctype html>
<meta charset="utf-8">
<title>YouTube → Whisper</title>
<style>
  body{max-width:840px;margin:40px auto;font:16px/1.4 system-ui;}
  input[type=text]{width:100%;padding:10px;font-size:16px;}
  button{padding:10px 16px;font-size:16px;margin-top:8px;cursor:pointer}
  pre{white-space:pre-wrap;border:1px solid #ddd;padding:12px;background:#fafafa}
  .row{margin:12px 0}
  .inline{display:inline-flex;gap:8px;align-items:center}
  .muted{color:#666;font-size:14px}
</style>
<h1>YouTube → MP3 → Whisper</h1>
<form method="post" action="/transcribe">
  <div class="row"><input name="youtube_url" type="text" placeholder="https://www.youtube.com/watch?v=..." required></div>
  <div class="row"><label>Language hint (optional): <input name="language" type="text" placeholder="zh, en, ja"></label></div>
  <div class="row"><button type="submit">Transcribe</button></div>
</form>

{% if result %}
  <h2>Result</h2>
  <p>Language: <b>{{ result.language }}</b></p>
  <p>Text length: {{ result.text_len }}</p>
  <ul>
    <li class="inline">
      <a id="txt-link" href="{{ result.text_url }}">Full Text (.txt)</a>
      <button id="copy-text" type="button" data-url="{{ result.text_url }}">Copy to clipboard</button>
      <span id="copy-status" class="muted" aria-live="polite"></span>
    </li>
    <li><a href="{{ result.srt_url }}">Subtitles (.srt)</a></li>
    <li><a href="{{ result.vtt_url }}">Subtitles (.vtt)</a></li>
    <li><a href="{{ result.segments_url }}">Segments (.json)</a></li>
    <li><a href="{{ result.mp3_url }}">Audio (.mp3)</a></li>
  </ul>
  <details>
    <summary>Preview</summary>
    <pre>{{ result.preview }}</pre>
  </details>
{% endif %}
<textarea id="clipboard-shadow" aria-hidden="true"
  style="position:fixed; left:-9999px; top:0; width:1px; height:1px; opacity:0; pointer-events:none;"></textarea>
<script>
(function(){
  const btn = document.getElementById('copy-text');
  if(!btn) return;
  const status = document.getElementById('copy-status');
  const url = btn.dataset.url;

  async function fetchText(u){
    const res = await fetch(u + (u.includes('?') ? '&' : '?') + '_ts=' + Date.now(), {cache:'no-store'});
    if(!res.ok) throw new Error('HTTP '+res.status);
    return await res.text();
  }

  function flash(msg, ok=true){
    status.textContent = msg;
    status.style.color = ok ? '#2b6a2b' : '#a33';
    clearTimeout(flash._t);
    flash._t = setTimeout(()=>{ status.textContent=''; }, 3000);
  }

  async function copyModern(text){
    if (!(navigator.clipboard && window.isSecureContext)) throw new Error('no modern api');
    await navigator.clipboard.writeText(text); // may throw NotAllowedError
  }

  function copyViaSelection(text){
    const ta = document.getElementById('clipboard-shadow');
    ta.value = text;
    // iOS/webkit often needs focus+select with a short sync layout pass
    ta.removeAttribute('readonly');
    ta.setSelectionRange(0, 0);
    ta.focus({preventScroll:true});
    ta.select();
    // execCommand returns boolean; check it
    const ok = document.execCommand('copy');
    ta.setAttribute('readonly','');
    ta.blur();
    ta.value = '';
    if (!ok) throw new Error('execCommand returned false');
  }

  // Last-resort UI for manual copy (shown only if both programmatic paths fail)
  function showManual(text){
    let dlg = document.getElementById('manual-copy');
    if(!dlg){
      dlg = document.createElement('div');
      dlg.id = 'manual-copy';
      dlg.style.position = 'fixed';
      dlg.style.inset = '0';
      dlg.style.background = 'rgba(0,0,0,0.3)';
      dlg.style.display = 'grid';
      dlg.style.placeItems = 'center';
      dlg.innerHTML = `
        <div style="background:#fff; max-width:800px; width:90%; padding:16px; border:1px solid #ddd">
          <div style="margin:0 0 8px 0; font-weight:600">Manual copy (Cmd/Ctrl+C)</div>
          <textarea id="manual-copy-ta" style="width:100%; height:300px"></textarea>
          <div style="margin-top:8px; display:flex; gap:8px; justify-content:flex-end">
            <button id="manual-select">Select all</button>
            <button id="manual-close">Close</button>
          </div>
        </div>`;
      document.body.appendChild(dlg);
      dlg.addEventListener('click', (e)=>{ if(e.target===dlg) dlg.remove(); });
      dlg.querySelector('#manual-close').addEventListener('click', ()=>dlg.remove());
      dlg.querySelector('#manual-select').addEventListener('click', ()=>{
        const mta = document.getElementById('manual-copy-ta');
        mta.focus();
        mta.select();
      });
    }
    const mta = dlg.querySelector('#manual-copy-ta');
    mta.value = text;
    document.body.appendChild(dlg);
    const selBtn = dlg.querySelector('#manual-select');
    dlg.style.display = 'grid';
    // auto-select for convenience
    selBtn.click();
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      const text = await fetchText(url);
      try {
        await copyModern(text);
        flash('Copied');
      } catch (_e1) {
        try {
          copyViaSelection(text);
          flash('Copied');
        } catch (_e2) {
          showManual(text);
          flash('Copy failed', false);
        }
      }
    } catch(e){
      console.error(e);
      flash('Copy failed', false);
    } finally {
      btn.disabled = false;
    }
  });
})();
</script>
""")

def _is_hf_repo_id(value: str) -> bool:
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", value))

def _check_ffmpeg_binary() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "`ffmpeg` executable was not found in PATH. Install ffmpeg and restart the app."
        )
    return ffmpeg_path

def _check_model_location() -> str:
    configured = (MODEL_PATH or "").strip()
    if not configured:
        raise RuntimeError("`MLX_WHISPER_MODEL` is empty.")

    resolved = os.path.abspath(os.path.expanduser(configured))
    if os.path.exists(resolved):
        return resolved
    if _is_hf_repo_id(configured):
        return configured

    raise RuntimeError(
        f"Model path is invalid: {configured}. Set `MLX_WHISPER_MODEL` to an existing local path "
        "or a Hugging Face repo id (for example: mlx-community/whisper-large-v3-turbo)."
    )

def _check_js_runtime() -> str:
    for name in JS_RUNTIMES:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "No JavaScript runtime found in PATH (needed by yt-dlp for YouTube). "
        "Install one of: " + ", ".join(JS_RUNTIMES) + " (e.g. `brew install deno`)."
    )

def _check_artifacts_dir() -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    probe_path = None
    fd = None
    try:
        fd, probe_path = tempfile.mkstemp(prefix=".write-check-", dir=ARTIFACTS_DIR)
    except OSError as e:
        raise RuntimeError(f"`{ARTIFACTS_DIR}` is not writable: {e}") from e
    finally:
        if fd is not None:
            os.close(fd)
        if probe_path and os.path.exists(probe_path):
            os.unlink(probe_path)
    return ARTIFACTS_DIR

def _run_environment_checks():
    checks = {}
    errors = []
    for key, checker in (
        ("ffmpeg", _check_ffmpeg_binary),
        ("model", _check_model_location),
        ("js_runtime", _check_js_runtime),
        ("artifacts", _check_artifacts_dir),
    ):
        try:
            checks[key] = {"ok": True, "detail": checker()}
        except Exception as e:
            msg = str(e)
            checks[key] = {"ok": False, "detail": msg}
            errors.append(f"{key}: {msg}")
    return checks, errors

@app.on_event("startup")
def validate_environment_on_startup():
    _, errors = _run_environment_checks()
    if errors:
        bullet_list = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(f"Startup checks failed:\n{bullet_list}")

def _safe_slug(s: str) -> str:
    # Normalize unicode
    s = unicodedata.normalize("NFKC", s)
    # Replace any filesystem-unsafe characters with underscore
    s = re.sub(r"[^\w\s\-().]", "_", s, flags=re.UNICODE)
    # Collapse spaces to underscores
    s = re.sub(r"\s+", "_", s)
    return s.strip("_") or "audio"

def _format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=float(seconds))
    # SRT hh:mm:ss,ms
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    h += td.days * 24
    ms = int(td.microseconds / 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _to_srt(segments: List[Dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_ts(seg.get("start", 0))
        end = _format_ts(seg.get("end", 0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def _to_vtt(segments: List[Dict]) -> str:
    out = ["WEBVTT\n"]
    for seg in segments:
        # VTT uses '.' for milliseconds
        def vtt_ts(t):
            ts = _format_ts(t).replace(",", ".")
            return ts
        start = vtt_ts(seg.get("start", 0))
        end = vtt_ts(seg.get("end", 0))
        text = seg.get("text", "").strip()
        out.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(out)

def _download_best_audio(youtube_url: str, tmpdir: str) -> str:
    # yt-dlp to bestaudio (m4a/webm), then we'll ffmpeg→mp3
    ydl_opts = {
        "outtmpl": os.path.join(tmpdir, "%(title)s.%(ext)s"),
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
        downloaded = ydl.prepare_filename(info)
        # If postprocessor changed extension, resolve
        if not os.path.exists(downloaded):
            # try common audio extensions
            base = os.path.splitext(downloaded)[0]
            for ext in (".m4a", ".webm", ".mp4", ".ogg", ".opus"):
                cand = base + ext
                if os.path.exists(cand):
                    downloaded = cand
                    break
        return downloaded

def _convert_to_mp3(src_audio_path: str, dst_mp3_path: str) -> None:
    # ffmpeg -i in -vn -ac 1 -ar 16000 -b:a 160k out.mp3
    # mono 16k helps some models; adjust if you prefer original sr
    stream = ffmpeg.input(src_audio_path)
    stream = ffmpeg.output(stream, dst_mp3_path, ac=1, ar=16000, audio_bitrate="160k", vn=None)
    ffmpeg.run(stream, quiet=True, overwrite_output=True)

def _transcribe(mp3_path: str, language_hint: str | None) -> Dict:
    kwargs = {}
    if language_hint:
        kwargs["language"] = language_hint
    return mlx_whisper.transcribe(mp3_path, path_or_hf_repo=MODEL_PATH, **kwargs)

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML.render(result=None)

@app.get("/healthz")
def healthz():
    checks, errors = _run_environment_checks()
    status_code = 200 if not errors else 503
    status = "ok" if not errors else "error"
    return JSONResponse({"status": status, "checks": checks}, status_code=status_code)

@app.post("/transcribe", response_class=HTMLResponse)
def transcribe_page(youtube_url: str = Form(...), language: str = Form(default="")):
    res = _process(youtube_url, language_hint=(language or None))
    return INDEX_HTML.render(result=res)

@app.post("/api/transcribe")
def transcribe_api(payload: Dict):
    youtube_url = payload.get("youtube_url") or ""
    language = payload.get("language") or None
    if not youtube_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    res = _process(youtube_url, language_hint=language)
    return JSONResponse(res)

def _process(youtube_url: str, language_hint: str | None):
    tmpdir = tempfile.mkdtemp(prefix="ytmp3_")
    try:
        src_audio = _download_best_audio(youtube_url, tmpdir)
        title = _safe_slug(os.path.splitext(os.path.basename(src_audio))[0]) or "audio"
        mp3_path = os.path.join(tmpdir, f"{title}.mp3")
        _convert_to_mp3(src_audio, mp3_path)

        result = _transcribe(mp3_path, language_hint=language_hint)
        text = result.get("text", "")
        segments = result.get("segments", [])
        language = result.get("language", "")

        # persist artifacts
        base = f"{title}"
        dst_mp3 = os.path.join(ARTIFACTS_DIR, base + ".mp3")
        dst_txt = os.path.join(ARTIFACTS_DIR, base + ".txt")
        dst_srt = os.path.join(ARTIFACTS_DIR, base + ".srt")
        dst_vtt = os.path.join(ARTIFACTS_DIR, base + ".vtt")
        dst_segments = os.path.join(ARTIFACTS_DIR, base + ".segments.json")

        shutil.copyfile(mp3_path, dst_mp3)
        with open(dst_txt, "w", encoding="utf-8") as f:
            f.write(text)
        with open(dst_srt, "w", encoding="utf-8") as f:
            f.write(_to_srt(segments))
        with open(dst_vtt, "w", encoding="utf-8") as f:
            f.write(_to_vtt(segments))
        with open(dst_segments, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        return {
            "language": language,
            "text_len": len(text),
            "preview": text[:1000],
            "text_url": f"/artifacts/{os.path.basename(dst_txt)}",
            "srt_url": f"/artifacts/{os.path.basename(dst_srt)}",
            "vtt_url": f"/artifacts/{os.path.basename(dst_vtt)}",
            "segments_url": f"/artifacts/{os.path.basename(dst_segments)}",
            "mp3_url": f"/artifacts/{os.path.basename(dst_mp3)}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
