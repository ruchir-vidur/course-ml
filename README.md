---
title: Vidur Course ML
emoji: 🧠
colorFrom: teal
colorTo: orange
sdk: docker
app_port: 7860
---

# Vidur Course ML

FastAPI service that answers questions about Vidur course transcripts. It retrieves relevant transcript chunks from Qdrant and streams responses from Groq.

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

## Hugging Face Spaces

This project is set up to run as a Docker Space.

Use these Space secrets and variables:

- `GROQ_API_KEY` as a secret
- `QDRANT_URL` as a secret or variable
- `QDRANT_API_KEY` as a secret
- `MONGO_URI` as a secret if you use the course catalog
- `MONGO_DB` as a variable if needed

Not needed on the Space runtime:

- `embedding.py`
- `migrate_qdrant.py`
- `qdrant_db/`
- local `.venv`

The app only needs the deployed code, your Qdrant Cloud connection, and the runtime secrets above.

## Move vectors to Qdrant Cloud

Qdrant Cloud has a free tier for testing and prototypes. At the time of writing,
it includes a single-node cluster with 0.5 vCPU, 1 GB RAM, and 4 GB disk. If the
collection grows beyond that, you need to move to a paid tier.

To use Cloud instead of the local `qdrant_db/` folder:

1. Create a Qdrant Cloud cluster and copy its URL and API key.
2. Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`.
3. Keep your current local `qdrant_db/` folder in place.
4. Run the migration script from `src`:

```powershell
Set-Location src
python migrate_qdrant.py
Set-Location ..
```

After migration, the API and embedding scripts will automatically use Cloud when
`QDRANT_URL` is present. If you want to keep using the local folder, leave those
variables unset.

## Deployment notes

This service uses a local Qdrant database, so the deployment must either build the index during its release process or attach persistent storage containing `qdrant_db/`. Configure secrets through the hosting provider's environment-variable settings; never commit `.env`.

Before exposing the API publicly, replace the permissive CORS setting in `src/api.py` with the exact frontend origin(s).

## Render Deployment

Render is the easiest fit for this backend now that Qdrant lives in Cloud.

Step by step:

1. Push this `course-ml` folder to a GitHub repo.
2. In Render, create a new Web Service and connect that repo.
3. Choose the repo root as the service root.
4. Use the included [render.yaml](render.yaml) or copy these values into Render's UI:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `cd src && uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add these environment variables in Render:
  - `GROQ_API_KEY`
  - `QDRANT_URL`
  - `QDRANT_API_KEY`
  - `MONGO_URI` if you want title resolution from Mongo
  - `MONGO_DB` if your Mongo database is not already in the URI
6. Deploy.

Things you do not need on Render now:

- `embedding.py`
- `migrate_qdrant.py`
- `qdrant_db/`
- `.venv/`
- Hugging Face Spaces settings

If you ever need to rebuild vectors, run `python src/embedding.py` locally and then migrate once with `python src/migrate_qdrant.py`.
