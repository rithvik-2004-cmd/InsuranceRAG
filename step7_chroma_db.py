
import json
import chromadb
from pathlib import Path

# Find the project folder
BASE_DIR = Path(__file__).resolve().parent

# Load existing chunks and embeddings
json_path = BASE_DIR / "insurance_embeddings.json"

with open(json_path, "r", encoding="utf-8") as file:
    records = json.load(file)

# Create a persistent Chroma database
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

# Create or connect to a collection
collection = client.get_or_create_collection(
    name="insurance_policies"
)

# Store each chunk, its embedding, and its text
collection.upsert(
    ids=[f"policy_chunk_{i}" for i in range(len(records))],
    documents=[record["text"] for record in records],
    embeddings=[record["embedding"] for record in records],
    metadatas=[
        {
            "source": "insurance_policy.txt",
            "chunk_number": i + 1
        }
        for i in range(len(records))
    ]
)

print("Chunks stored successfully!")

print("Total records:", collection.count())
print("Database location:", BASE_DIR / "chroma_db")
