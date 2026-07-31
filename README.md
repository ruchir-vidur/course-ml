# Vidur Course ML

FastAPI service that answers questions about Vidur course transcripts. It retrieves relevant transcript chunks from a local Qdrant vector index and streams responses from Groq.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Course transcripts in WebVTT (`.vtt`) format, if you need to build or rebuild the local index

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. `MONGO_URI` is optional; it enables resolving MongoDB course IDs to the corresponding transcript folders.

## Prepare course data

Put each course's `.vtt` sessions in a separate folder under `data/`, for example:

```text
data/
  Example Course/
    session 1.vtt
    session 2.vtt
```

Build the local Qdrant index from the `src` directory:

```powershell
Set-Location src
python embedding.py
Set-Location ..
```

The transcripts and generated `qdrant_db/` are deliberately excluded from Git. Only add them to a private repository if you have permission to distribute them.

## Run the API

```powershell
Set-Location src
uvicorn api:app --host 0.0.0.0 --port 8000
```

Available endpoints include:

- `GET /health`
- `GET /courses`
- `POST /chat` (streaming response)

For development, use `--reload` with the Uvicorn command above.

## Deployment notes

This service uses a local Qdrant database, so the deployment must either build the index during its release process or attach persistent storage containing `qdrant_db/`. Configure secrets through the hosting provider's environment-variable settings; never commit `.env`.

Before exposing the API publicly, replace the permissive CORS setting in `src/api.py` with the exact frontend origin(s).
