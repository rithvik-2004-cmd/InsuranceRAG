
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    client.delete_collection(name="insurance_policies")
    print("Old insurance_policies collection deleted.")
except Exception:
    print("Collection was not found.")

collection = client.create_collection(
    name="insurance_policies",
    metadata={"hnsw:space": "l2"},
)

print("New collection created.")
print("Records currently stored:", collection.count())
