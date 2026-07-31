"""Copy the existing local Qdrant collection into Qdrant Cloud.

Usage:
  1. Set QDRANT_URL and QDRANT_API_KEY in course-ml/.env.
  2. Keep QDRANT_LOCAL_PATH pointed at the current local qdrant_db/ folder.
  3. Run: python migrate_qdrant.py
"""

import os

from qdrant_client.http import models as qmodels

from qdrant_connection import create_qdrant_client

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass


def _collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", "vidur_course_embeddings")


def migrate(batch_size: int = 256) -> None:
    collection_name = _collection_name()

    if os.environ.get("QDRANT_URL") is None:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in .env before running the migration.")

    try:
        local_client = create_qdrant_client(force_local=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Local qdrant_db is already locked by another process. Close any running API or Qdrant client and try again."
        ) from exc

    cloud_client = create_qdrant_client()

    try:
        source_info = local_client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001 - surface the exact Qdrant failure
        raise RuntimeError(f"Local collection {collection_name!r} was not found.") from exc

    source_vectors = source_info.config.params.vectors

    try:
        cloud_client.get_collection(collection_name)
    except Exception:
        cloud_client.create_collection(
            collection_name=collection_name,
            vectors_config=source_vectors,
        )

    total = 0
    offset = None
    while True:
        points, offset = local_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        cloud_client.upsert(
            collection_name=collection_name,
            points=[
                qmodels.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
        )
        total += len(points)
        print(f"Migrated {total} points...")

        if offset is None:
            break

    print(f"Done. Migrated {total} points into Qdrant Cloud collection {collection_name!r}.")


if __name__ == "__main__":
    migrate()