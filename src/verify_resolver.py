"""Ad-hoc check: connect to Mongo, list courses (id + title), and show how each
resolves to an indexed course folder. Run: python verify_resolver.py
"""
import os
from course_resolver import (
    CourseResolver,
    list_indexed_courses,
    load_mongo_id_to_title,
    load_aliases,
)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
indexed = list_indexed_courses(data_dir)
print("Indexed course folders:")
for c in indexed:
    print("  -", c)

id_to_title = load_mongo_id_to_title(os.environ.get("MONGO_URI"), os.environ.get("MONGO_DB"))
aliases = load_aliases(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "course_aliases.json"))
resolver = CourseResolver(indexed, id_to_title, aliases)

print("\nMongo _id | title | resolved:")
for _id, title in sorted(id_to_title.items(), key=lambda kv: kv[1].lower()):
    resolved = resolver.resolve(title)
    hit = "OK  " if resolved in indexed else "MISS"
    print(f"  [{hit}] {_id}  {title!r}  ->  {resolved!r}")
