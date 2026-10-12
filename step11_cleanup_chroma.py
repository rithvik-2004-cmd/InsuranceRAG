
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_collection(
    name="insurance_policies"
)

# Find only the records from the old test document.
old_records = collection.get(
    where={"source": "insurance_policy.txt"},
    include=["metadatas"]
)

old_ids = old_records["ids"]

print("Old test records found:", len(old_ids))

if old_ids:
    collection.delete(ids=old_ids)
    print("Old test records deleted.")

print("Remaining records:", collection.count())
