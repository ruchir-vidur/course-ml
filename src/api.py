import os
import re
import json
import threading
from contextlib import asynccontextmanager

from qdrant_client.http import models as qdrant_models
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from course_resolver import (
    CourseResolver,
    list_indexed_courses,
    load_mongo_id_to_title,
    load_aliases,
)
from qdrant_connection import create_qdrant_client, qdrant_mode_label

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
# Retrieve a wider candidate set, then a cross-encoder reranks down to the best
# few — noticeably better than raw vector top-k, still light (ONNX, no torch).
SIMILARITY_TOP_K = int(os.environ.get("SIMILARITY_TOP_K", "8"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "4"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
# MongoDB (course catalog) — used to build an accurate id→title map so the server
# can resolve whatever the frontend sends to the right indexed course.
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")  # optional; defaults to the DB in the URI

def _qa_prompt(course_name: str) -> PromptTemplate:
    """Course-aware QA prompt. Produces clean Markdown grounded strictly in the
    transcript context. It must NOT paste URLs or citation lists — the app renders
    the exact source videos as clickable chips from structured metadata."""
    return PromptTemplate(
        'You are Vidur, a warm and precise AI tutor for the course "' + course_name + '". '
        "Help the learner using ONLY the course transcript excerpts provided below.\n\n"
        "How to write your answer:\n"
        "- Answer in clean Markdown. Lead with the direct answer, then add supporting detail.\n"
        "- Use short paragraphs, and bullet or numbered lists for steps or multiple points. **Bold** key terms.\n"
        "- Be concise and warm, like a great teacher — no filler, no repetition, no preamble like 'Based on the context'.\n\n"
        "Grounding rules (important):\n"
        "- Base every statement strictly on the context. Never use outside knowledge or invent names, facts, or numbers.\n"
        "- Refer to specific teachings naturally in prose when useful (e.g., \"In Session 2, the teacher explains…\").\n"
        "- Do NOT paste URLs, links, or a list of citations — the app shows the exact source videos separately, below your answer.\n"
        "- If the answer isn't in the context, say so briefly in one sentence and point to what this course does cover that's related.\n\n"
        "Course context:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Learner's question: {query_str}\n\n"
        "Write your Markdown answer. Then, on the very last line, output a JSON array of exactly 3 short, "
        "specific follow-up questions the learner is likely to ask next about this course — "
        'e.g. ["...", "...", "..."]. Output nothing after that array.\n\n'
        "Your answer:"
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


def _ensure_course_filter_index(client) -> None:
    """Create the payload index required for filtering chunks by course_id."""
    collection_info = client.get_collection(COLLECTION)
    if "course_id" in (collection_info.payload_schema or {}):
        return

    print("   Creating Qdrant keyword index for course_id...")
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="course_id",
        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("1. Loading embedding model...")
    Settings.embed_model = FastEmbedEmbedding(model_name=EMBED_MODEL)

    print("2. Connecting to Groq...")
    Settings.llm = Groq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    print("3. Connecting to Qdrant...")
    client = create_qdrant_client()
    qdrant_mode = qdrant_mode_label()
    print(f"   Using {qdrant_mode} Qdrant backend.")
    # Embedded/local Qdrant supports filtering without payload indexes. A Qdrant
    # server requires this index and otherwise rejects the course_id filter.
    if qdrant_mode == "cloud":
        _ensure_course_filter_index(client)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)

    print("4. Loading index...")
    _state["index"] = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    print("5. Building course resolver...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
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


# Lazily-loaded cross-encoder reranker (FastEmbed / ONNX). Loaded on first use so
# startup stays fast; if it can't load or run, we fall back to plain vector order.
_reranker: dict = {"enc": None, "failed": False}


def _rerank(query: str, nodes: list, top_n: int) -> list:
    """Reorder retrieved nodes by cross-encoder relevance and keep the best top_n.
    Never raises — on any failure it returns the vector-ordered top_n. Set the env
    var RERANK_MODEL to an empty string to disable reranking entirely."""
    if not nodes or _reranker["failed"] or not RERANK_MODEL:
        return nodes[:top_n]
    try:
        if _reranker["enc"] is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            print(f"   Loading reranker {RERANK_MODEL}...")
            _reranker["enc"] = TextCrossEncoder(model_name=RERANK_MODEL)
        docs = [n.node.get_content() for n in nodes]
        scores = list(_reranker["enc"].rerank(query, docs))
        ranked = sorted(zip(nodes, scores), key=lambda p: p[1], reverse=True)
        for n, s in ranked:
            n.score = float(s)
        return [n for n, _ in ranked[:top_n]]
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️  Reranker unavailable, using vector order ({e}).")
        _reranker["failed"] = True
        return nodes[:top_n]


def _retrieve(course_id: str, message: str) -> list:
    """Vector retrieve (filtered to the course), then cross-encoder rerank."""
    course_filter = MetadataFilters(
        filters=[ExactMatchFilter(key="course_id", value=course_id)]
    )
    retriever = _state["index"].as_retriever(
        similarity_top_k=SIMILARITY_TOP_K, filters=course_filter
    )
    # Retrieval touches Qdrant — serialize it (local file-based Qdrant is single-writer).
    with _retrieval_lock:
        nodes = retriever.retrieve(message)
    return _rerank(message, nodes, RERANK_TOP_N)


def _pretty_session(session: str | None) -> str:
    """'session 3' / 'session3' -> 'Session 3' for nicer source-chip labels."""
    if not session:
        return "Source"
    m = re.match(r"\s*session\s*(\d+)", session, re.IGNORECASE)
    if m:
        return f"Session {m.group(1)}"
    return session[:1].upper() + session[1:]


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
        # A short transcript preview so the learner sees why this moment was cited.
        try:
            text = re.sub(r"\s+", " ", (node.get_content() or "")).strip()
        except Exception:
            text = ""
        snippet = text[:160].rstrip() + ("…" if len(text) > 160 else "")
        resources.append(
            {
                # `title` + `url` match the workspace ChatResource shape so the
                # frontend renders these as clickable source links directly.
                "title": _pretty_session(session),
                "url": link,
                "session": session,
                "video_url": video_url,
                "start_time_seconds": start,
                "snippet": snippet,
            }
        )
    return resources


def event_stream(course_identifier: str, message: str):
    try:
        # Resolve whatever the frontend sent (title or hash id) to the exact
        # course_id the index is keyed on.
        resolver: CourseResolver = _state["resolver"]
        course_id = resolver.resolve(course_identifier) if resolver else course_identifier

        # Retrieve a wide candidate set, rerank to the best few, then synthesize a
        # streamed answer grounded in exactly those chunks.
        nodes = _retrieve(course_id, message)
        synthesizer = get_response_synthesizer(
            text_qa_template=_qa_prompt(course_id),
            response_mode="compact",
            streaming=True,
        )
        response = synthesizer.synthesize(message, nodes=nodes)

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

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
