import os
import json
import threading
from contextlib import asynccontextmanager

import qdrant_client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from course_resolver import (
    CourseResolver,
    list_indexed_courses,
    load_mongo_id_to_title,
    load_aliases,
)

# Load a local course-ml/.env if present (for GROQ_API_KEY, MONGO_URI, etc.).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass

# --- Config -----------------------------------------------------------------
# Read from the environment (loaded from course-ml/.env above). Nothing secret is
# hardcoded — set GROQ_API_KEY in .env.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to course-ml/.env (see .env.example).")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "vidur_course_embeddings")
SIMILARITY_TOP_K = int(os.environ.get("SIMILARITY_TOP_K", "3"))
# MongoDB (course catalog) — used to build an accurate id→title map so the server
# can resolve whatever the frontend sends to the right indexed course.
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")  # optional; defaults to the DB in the URI

QA_PROMPT_TMPL = PromptTemplate(
    "You are an expert AI teaching assistant for a course. \n"
    "Below is context from the course transcripts. Each piece of context has metadata attached.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the user's query.\n"
    "You MUST explicitly cite the exact session and provide the video link at the end of your answer.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Local (file-based) Qdrant locks the DB to a single client, and the sqlite
# backend is not safe under concurrent access from multiple threads. Because
# StreamingResponse iterates our sync generator in a worker thread, we guard the
# retrieval step (the part that actually touches Qdrant) with a lock. Token
# streaming from Groq happens outside the lock, so users don't block each other
# while the model is generating.
_retrieval_lock = threading.Lock()

# Populated at startup.
_state: dict = {"index": None, "resolver": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("1. Loading embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    print("2. Connecting to Groq...")
    Settings.llm = Groq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    print("3. Connecting to Qdrant...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "..", "qdrant_db")
    client = qdrant_client.QdrantClient(path=db_path)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)

    print("4. Loading index...")
    _state["index"] = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    print("5. Building course resolver...")
    data_dir = os.path.join(current_dir, "..", "data")
    indexed_courses = list_indexed_courses(data_dir)
    id_to_title = load_mongo_id_to_title(MONGO_URI, MONGO_DB)
    aliases = load_aliases(os.path.join(current_dir, "..", "course_aliases.json"))
    _state["resolver"] = CourseResolver(indexed_courses, id_to_title, aliases)
    print(f"   Indexed courses: {indexed_courses}")
    if aliases:
        print(f"   Loaded {len(aliases)} course alias override(s).")
    print("✅ Ready.")

    yield

    # Release the local Qdrant lock on shutdown.
    try:
        client.close()
    except Exception:
        pass


app = FastAPI(title="Vidur Course AI", lifespan=lifespan)

# Allow the frontend to call this directly. Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    # The frontend sends the course TITLE here; a backend hash _id also works.
    # `course_title` is accepted as an explicit alias. The server resolves either
    # to the exact indexed course_id (the transcript folder name).
    course_id: str | None = None
    course_title: str | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _build_query_engine(course_id: str):
    course_filter = MetadataFilters(
        filters=[ExactMatchFilter(key="course_id", value=course_id)]
    )
    return _state["index"].as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        filters=course_filter,
        text_qa_template=QA_PROMPT_TMPL,
        streaming=True,
    )


def _resources_from_response(response) -> list:
    """Deduplicated list of source videos (session + deep-link with timestamp)."""
    seen = set()
    resources = []
    for node in getattr(response, "source_nodes", []) or []:
        md = node.metadata or {}
        video_url = md.get("video_url")
        start = md.get("start_time_seconds")
        session = md.get("session")
        if not video_url:
            continue
        link = f"{video_url}?t={start}" if start is not None else video_url
        if link in seen:
            continue
        seen.add(link)
        resources.append(
            {
                # `title` + `url` match the workspace ChatResource shape so the
                # frontend renders these as clickable source links directly.
                "title": session or "Source",
                "url": link,
                "session": session,
                "video_url": video_url,
                "start_time_seconds": start,
            }
        )
    return resources


def event_stream(course_identifier: str, message: str):
    try:
        # Resolve whatever the frontend sent (title or hash id) to the exact
        # course_id the index is keyed on.
        resolver: CourseResolver = _state["resolver"]
        course_id = resolver.resolve(course_identifier) if resolver else course_identifier

        engine = _build_query_engine(course_id)

        # Retrieval touches local Qdrant — serialize it.
        with _retrieval_lock:
            response = engine.query(message)

        # Stream the LLM tokens as they arrive.
        for token in response.response_gen:
            if token:
                yield _sse({"type": "token", "content": token})

        # After the answer, emit the source videos + the resolved course as metadata.
        yield _sse(
            {
                "type": "metadata",
                "course_id": course_id,
                "resources": _resources_from_response(response),
            }
        )
        yield _sse({"type": "done"})
    except Exception as e:  # noqa: BLE001 - surface any failure to the client
        yield _sse({"type": "error", "content": str(e)})


@app.get("/health")
def health():
    return {"status": "ok", "ready": _state["index"] is not None}


@app.get("/courses")
def available_courses():
    """Mongo courses that actually have vectorized transcripts — i.e. whose title
    (or alias) resolves to an indexed transcript folder. The workspace uses this
    to only offer "Chat about this course" where it will actually work.
    """
    resolver: CourseResolver = _state["resolver"]
    if not resolver:
        return {"courses": []}
    out = []
    for _id, title in resolver.id_to_title.items():
        folder = resolver.resolve(title)
        if folder in resolver.indexed_courses:
            out.append({"id": _id, "title": title, "course_id": folder})
    return {"courses": out}


@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    identifier = req.course_title or req.course_id or ""
    return StreamingResponse(
        event_stream(identifier, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering (e.g. nginx) so tokens flush immediately.
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
