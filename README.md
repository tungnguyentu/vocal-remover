# Vocal Remover

Local webapp to remove vocals from video/audio using Apple Silicon GPU (MPS).

## Stack

- **demucs** (htdemucs) — vocal separation via MPS
- **FastAPI + uvicorn** — backend
- **ffmpeg** — audio extraction

## Setup

```bash
pip install fastapi uvicorn aiofiles python-multipart
pip install demucs
```

## Run

```bash
cd webapp
uvicorn main:app --reload
```

Open http://localhost:8000, drop a video/audio file, click **Remove Vocals (GPU)**.

## Output

- `vocals.wav` — isolated vocals
- `no_vocals.wav` — instrumental track
