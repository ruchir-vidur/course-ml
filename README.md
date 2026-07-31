# Vidur Course ML

FastAPI service that answers questions about Vidur course transcripts. It embeds
the query with **FastEmbed** (ONNX — no torch), retrieves the relevant transcript
chunks from **Qdrant Cloud**, and streams the answer from **Groq**.

## Requirements

- Python 3.11 (pinned in `.python-version`)
- A Groq API key
- A Qdrant Cloud cluster (URL + API key)
- Optionally, MongoDB (to resolve course `_id`s / titles to transcript folders)

## Endpoints

- `GET /health` — readiness
- `GET /courses` — courses that have vectorized transcripts
- `POST /chat_stream` — streaming (SSE) answer; body: `{ "course_title": "...", "message": "..." }`
  (`course_id` is also accepted; the server resolves a title or hash id to the
  right transcript folder)

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` (see `.env.example`): `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
and optionally `MONGO_URI` (+ `MONGO_DB`). Without `QDRANT_URL` the app falls back
to a local on-disk `qdrant_db/` folder for development.

## Prepare course data + build the index

Put each course's `.vtt` sessions in a folder under `data/` (the folder name is
the course's `course_id`):

```text
data/
  Example Course/
    session 1.vtt
    session 2.vtt
```

With `QDRANT_URL`/`QDRANT_API_KEY` set, build the index straight into Qdrant Cloud:

```powershell
Set-Location src
python embedding.py      # embeds with FastEmbed and upserts to Qdrant Cloud
Set-Location ..
```

The transcripts and any local `qdrant_db/` are excluded from Git — only add them
to a private repo if you're allowed to distribute them.

> Note: the embedding backend must match between indexing and querying. Both use
> FastEmbed (`BAAI/bge-small-en-v1.5`), so always (re)build the index with
> `embedding.py` — don't mix in vectors produced by a different embedder.

## Run the API locally

```powershell
Set-Location src
uvicorn api:app --host 0.0.0.0 --port 8000   # add --reload for development
```

## Deploy on Render

Render is the intended host now that Qdrant lives in Cloud.

1. Push this `course-ml` folder to a GitHub repo (repo root = `course-ml`).
2. In Render, create a **Web Service** from that repo. It reads `render.yaml`, or set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `cd src && uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Health Check Path: `/health`
3. Python 3.11.8 is pinned via `.python-version` — no extra config needed.
4. Add environment variables:
   - `GROQ_API_KEY`
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `MONGO_URI` (optional — enables Mongo title resolution), `MONGO_DB` (optional)
5. Deploy.

The image is intentionally light (FastEmbed/ONNX, no torch), so it fits Render's
smaller instances. First boot downloads the small ONNX embedding model (~130MB).

Only the runtime needs to reach Qdrant Cloud + Groq + (optionally) Mongo — you do
**not** need `qdrant_db/` or `.venv/` on the host. `embedding.py` is only used to
build the index; it isn't part of serving requests.

## Before going public

Replace the permissive CORS setting in `src/api.py` with the exact frontend
origin(s), and never commit `.env`.
