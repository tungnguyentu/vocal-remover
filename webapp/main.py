import asyncio
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import mlx_whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

_thread_pool = ThreadPoolExecutor(max_workers=1)

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

app = FastAPI()
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

# Track job status in memory (good enough for local use)
jobs: dict[str, dict] = {}


async def run(cmd: list[str], job_id: str, step: str):
    jobs[job_id]["status"] = step
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = out.decode()[-500:]
        raise RuntimeError(out.decode()[-500:])
    return out.decode()



@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE / "index.html").read_text()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename).suffix
    src = UPLOADS / f"{job_id}{suffix}"

    async with aiofiles.open(src, "wb") as f:
        await f.write(await file.read())

    jobs[job_id] = {"status": "uploaded", "filename": file.filename}
    return {"job_id": job_id}


@app.post("/process/{job_id}")
async def process(
    job_id: str,
    extract_audio: bool = True,
    remove_vocals: bool = True,
    transcribe: bool = False,
    language: str = "zh",
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    asyncio.create_task(_pipeline(job_id, extract_audio, remove_vocals, transcribe, language))
    return {"job_id": job_id, "status": "started"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    import re
    if not re.fullmatch(r"[A-Za-z0-9._-]+", job_id) or not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        raise HTTPException(404, "File not found")
    base = OUTPUTS.resolve()
    try:
        path = (OUTPUTS / job_id / filename).resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(404, "File not found")
    if base not in path.parents:
        raise HTTPException(404, "File not found")
    return FileResponse(path)


def _run_mlx_whisper(audio_path: str, language: str, job_id: str) -> dict:
    """Runs mlx-whisper synchronously (called in thread pool). Updates progress per segment."""
    # mlx-whisper large-v3 in MLX format; downloaded to HF cache on first run
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language=language,
        word_timestamps=False,
        verbose=False,
        # segment callback for live progress
        condition_on_previous_text=True,
    )
    # update progress after each segment by post-processing the result segments
    segments = result.get("segments", [])
    duration = segments[-1]["end"] if segments else 1
    for seg in segments:
        pct = min(int(seg["end"] / duration * 100), 99)
        ts = f"{_fmt(seg['start'])} --> {_fmt(seg['end'])}"
        jobs[job_id]["progress"] = f"{pct}% — {ts} — {seg['text'].strip()[:60]}"
    return result


def _fmt(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def _write_srt(segments: list, path: Path):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt(seg['start'])} --> {_fmt(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def _pipeline(job_id: str, extract_audio: bool, remove_vocals: bool, transcribe: bool, language: str):
    try:
        src = next(UPLOADS.glob(f"{job_id}.*"))
        out_dir = OUTPUTS / job_id
        out_dir.mkdir(exist_ok=True)

        audio_path = src

        if extract_audio and src.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            wav = out_dir / "audio.wav"
            await run(
                ["ffmpeg", "-y", "-i", str(src), "-ar", "44100", "-ac", "2", str(wav)],
                job_id, "extracting_audio"
            )
            audio_path = wav
            jobs[job_id]["audio"] = f"/outputs/{job_id}/audio.wav"

        if remove_vocals:
            await run(
                [
                    "nice", "-n", "10",
                    "demucs", "--two-stems=vocals",
                    "--device", "mps",
                    "-o", str(out_dir),
                    str(audio_path),
                ],
                job_id, "removing_vocals"
            )
            # demucs outputs to out_dir/htdemucs/<stem>/
            stem_dir = out_dir / "htdemucs" / audio_path.stem
            no_vocals = stem_dir / "no_vocals.wav"
            vocals = stem_dir / "vocals.wav"
            if no_vocals.exists():
                jobs[job_id]["no_vocals"] = f"/outputs/{job_id}/htdemucs/{audio_path.stem}/no_vocals.wav"
            if vocals.exists():
                jobs[job_id]["vocals"] = f"/outputs/{job_id}/htdemucs/{audio_path.stem}/vocals.wav"

        if transcribe:
            transcribe_src = audio_path
            srt = out_dir / "transcript.srt"
            jobs[job_id]["status"] = "transcribing"
            jobs[job_id]["progress"] = "loading model…"

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _thread_pool,
                lambda: _run_mlx_whisper(str(transcribe_src), language, job_id),
            )

            # write SRT
            _write_srt(result["segments"], srt)
            if srt.exists():
                jobs[job_id]["transcript"] = f"/outputs/{job_id}/transcript.srt"

        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)[:500]
