import asyncio
import re
import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"

app = FastAPI()
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

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
async def process(job_id: str, extract_audio: bool = True, remove_vocals: bool = True,
                  transcribe: bool = False, language: str = "zh"):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    asyncio.create_task(_pipeline(job_id, extract_audio, remove_vocals))
    return {"job_id": job_id, "status": "started"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
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


async def _pipeline(job_id: str, extract_audio: bool, remove_vocals: bool):
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
                    "--device", "cpu",
                    "-o", str(out_dir),
                    str(audio_path),
                ],
                job_id, "removing_vocals"
            )
            stem_dir = out_dir / "htdemucs" / audio_path.stem
            no_vocals = stem_dir / "no_vocals.wav"
            if no_vocals.exists():
                jobs[job_id]["no_vocals"] = f"/outputs/{job_id}/htdemucs/{audio_path.stem}/no_vocals.wav"

        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)[:500]
