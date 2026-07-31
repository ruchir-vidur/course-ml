"""Resolve whatever the frontend sends (a course title, or a backend hash _id)
to the exact course_id the RAG index is keyed on — i.e. the transcript folder
name used by parser.py.

Two inputs feed the resolver:
  • The set of indexed course names: the sub-folder names under ``data/`` (parser.py
    stores each as ``course_id`` on every chunk). This is the ground truth of what
    we can actually answer about.
  • A ``{_id -> title}`` map pulled live from MongoDB, so an incoming hash id can be
    translated to its title, and titles stay accurate to the real course catalog.

Matching is forgiving: exact after normalization, then a fuzzy fallback, so small
differences (punctuation, spacing, casing) between the Mongo title and the folder
name still resolve.
"""

import os
import re
import difflib


def normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def list_indexed_courses(data_dir: str) -> list[str]:
    """The course_id values in the index = immediate sub-folders of ``data/``."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


def load_mongo_id_to_title(mongo_uri: str | None, db_name: str | None = None) -> dict:
    """Return ``{str(_id): title}`` from the courses collection.

    Never raises: on any connection/credential/network failure it logs and returns
    an empty map, so the chat server still runs on title-only matching.
    """
    if not mongo_uri:
        print("   MONGO_URI not set — skipping Mongo course map (title-only matching).")
        return {}
    try:
        from pymongo import MongoClient

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name] if db_name else client.get_default_database()
        if db is None:  # URI had no default DB and none supplied
            db = client["sattvastha"]
        mapping = {
            str(doc["_id"]): doc.get("title", "")
            for doc in db["courses"].find({}, {"title": 1})
            if doc.get("title")
        }
        client.close()
        print(f"   Loaded {len(mapping)} courses from MongoDB ({db.name}.courses).")
        return mapping
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️  Could not load courses from MongoDB ({e}) — title-only matching.")
        return {}


def load_aliases(path: str) -> dict:
    """Load explicit ``{ mongo _id (or title) -> folder name }`` overrides.

    For courses whose Mongo title doesn't match (or fuzzy-match) the transcript
    folder name, pin the pairing here. Missing/blank values are ignored. Never
    raises — a missing/invalid file just means no overrides.
    """
    import json

    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Keys starting with "_" are notes/comments, not aliases.
        return {
            k: v
            for k, v in data.items()
            if not k.startswith("_") and isinstance(v, str) and v.strip()
        }
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️  Could not read course aliases ({e}).")
        return {}


class CourseResolver:
    def __init__(
        self,
        indexed_courses: list[str],
        id_to_title: dict,
        aliases: dict | None = None,
        cutoff: float = 0.6,
    ):
        self.indexed_courses = indexed_courses
        self.id_to_title = id_to_title
        self.aliases = aliases or {}
        self.cutoff = cutoff
        # normalized folder name -> real folder name
        self._norm_to_course = {normalize(c): c for c in indexed_courses}
        # Aliases can be keyed by Mongo _id or by title. Build a normalized-title
        # index so an incoming title (what the frontend sends) hits its override.
        self._alias_norm: dict[str, str] = {}
        for key, folder in self.aliases.items():
            self._alias_norm[normalize(key)] = folder
            title = self.id_to_title.get(key)
            if title:
                self._alias_norm[normalize(title)] = folder

    def resolve(self, identifier: str) -> str:
        """Map a title or hash _id to the exact indexed course_id.

        Falls back to returning the input unchanged when nothing matches (the
        query then simply retrieves no context, rather than erroring).
        """
        raw = (identifier or "").strip()
        if not raw:
            return raw

        # 0) explicit alias keyed by the raw id/title.
        if raw in self.aliases:
            return self.aliases[raw]

        # If it's a backend hash id, translate to the course title first.
        candidate = self.id_to_title.get(raw, raw)
        n = normalize(candidate)

        # 1) explicit alias keyed by (normalized) title.
        if n in self._alias_norm:
            return self._alias_norm[n]
        # 2) exact match against the indexed folder names.
        if n in self._norm_to_course:
            return self._norm_to_course[n]
        # 3) fuzzy match against the indexed folder names.
        close = difflib.get_close_matches(n, list(self._norm_to_course.keys()), n=1, cutoff=self.cutoff)
        if close:
            return self._norm_to_course[close[0]]
        # 4) give up gracefully — return the (title) candidate as-is.
        return candidate
