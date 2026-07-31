import os
from pathlib import Path

import qdrant_client


def _default_local_path() -> str:
    return str((Path(__file__).resolve().parent / ".." / "qdrant_db").resolve())


def qdrant_mode_label() -> str:
    return "cloud" if os.environ.get("QDRANT_URL") else "local"


def create_qdrant_client(force_local: bool = False) -> qdrant_client.QdrantClient:
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    local_path = os.environ.get("QDRANT_LOCAL_PATH") or _default_local_path()

    if not force_local and qdrant_url:
        if not qdrant_api_key:
            raise RuntimeError("QDRANT_URL is set but QDRANT_API_KEY is missing.")
        return qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    return qdrant_client.QdrantClient(path=local_path)