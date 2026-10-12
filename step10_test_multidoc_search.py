
import os
import requests
import chromadb
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

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
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": [question],
            "input_type": "query",
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])
    return data[0]["embedding"]


questions = [
    "Who receives the death benefit in life insurance?",
    "What should I do after a car accident?",
    "What documents may be needed for an insurance claim?",
    "What is a deductible in health insurance?",
    "What is an insurance premium?",
]

for question in questions:
    print("\n" + "=" * 65)
    print("QUESTION:", question)

    query_embedding = get_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    for rank, document in enumerate(results["documents"][0], start=1):
        metadata = results["metadatas"][0][rank - 1]
        distance = results["distances"][0][rank - 1]

        print(f"\nResult {rank} | Distance: {distance:.4f}")
        print("Source:", metadata["source"])
        print("Type:", metadata.get("document_type", "Original test data"))
        print("Text:", document[:350])
