
import os
import json
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY")

EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

# Connect to the existing Chroma database
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_collection(
    name="insurance_policies"
)


def get_query_embedding(question):
    response = requests.post(
        EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": [question],
            "input_type": "query"
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])

    return data[0]["embedding"]


question = input("Ask an insurance question: ")

query_embedding = get_query_embedding(question)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, collection.count()),
    include=["documents", "metadatas", "distances"]
)

print("\nRetrieved policy sections:\n")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(f"--- Result {i + 1} ---")
    print("Source:", metadata["source"])
    print("Chunk number:", metadata["chunk_number"])
    print(f"Distance: {distance:.4f}")
    print("Text:", document)
    print()
