
import chromadb
import requests
import os

from pathlib import Path
from dotenv import load_dotenv


# 1. Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY")


# 2. NVIDIA embedding configuration
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"


# 3. Connect to the existing ChromaDB
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_collection(
    name="insurance_policies"
)

print("Environment loaded:", bool(API_KEY))
print("Documents stored:", collection.count())


# 4. Convert a question into a NVIDIA embedding
def get_query_embedding(question):
    """Convert the user's question into a vector."""

    if not API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing from .env")

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
            "encoding_format": "float",
        },
        timeout=60,
    )

    response.raise_for_status()

    embedding = response.json()["data"][0]["embedding"]

    if len(embedding) != 2048:
        raise ValueError(
            f"Expected 2048 dimensions, got {len(embedding)}"
        )

    return embedding


# 5. Evaluation dataset
evaluation_questions = [
    {
        "question": "What is a policy limit?",
        "relevant_sections": {
            "What Is a Policy Limit?",
            "Policy Limit",
        },
    },
    {
        "question": "What documents are needed for an insurance claim?",
        "relevant_sections": {
            "Supporting Documents",
            "Claim Process",
            "Theft",
        },
    },
    {
        "question": "What should I do if my insured car is stolen?",
        "relevant_sections": {
            "Theft",
            "Claim Notification",
        },
    },
    {
        "question": "What expenses are covered during hospitalization?",
        "relevant_sections": {
            "Hospitalization Coverage",
            "Exclusions",
            "Deductible",
            "Policy Limit",
        },
    },
    {
        "question": "Who receives the death benefit in life insurance?",
        "relevant_sections": {
            "Death Benefit",
            "Purpose of Life Insurance",
        },
    },
]


# 6. Evaluate retrieval for each question
scores = []

for test in evaluation_questions:
    question = test["question"]
    relevant_sections = test["relevant_sections"]

    # Convert question into an embedding
    query_embedding = get_query_embedding(question)

    # Retrieve the top 3 sections from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_metadata = results["metadatas"][0]
    retrieved_documents = results["documents"][0]
    distances = results["distances"][0]

    # Print question
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("Retrieved section contents:")

    # Display each retrieved section
    for metadata, document, distance in zip(
        retrieved_metadata,
        retrieved_documents,
        distances,
    ):
        print(
            f"\n{metadata.get('source', 'Unknown')} | "
            f"{metadata.get('section_title', 'Unknown')} | "
            f"Distance: {distance:.3f}"
        )

        print(document[:500])
        print("-" * 40)

    # Calculate Precision@3 using exact section-title matching
    relevant_count = sum(
        metadata.get("section_title", "Unknown") in relevant_sections
        for metadata in retrieved_metadata
    )

    total_retrieved = len(retrieved_metadata)

    precision = (
        relevant_count / total_retrieved
        if total_retrieved
        else 0
    )

    scores.append(precision)

    print(
        f"Relevant sections: "
        f"{relevant_count}/{total_retrieved}"
    )
    print(f"Precision@3: {precision:.2%}")


# 7. Calculate the average score
average_precision = (
    sum(scores) / len(scores)
    if scores
    else 0
)

print("\n========== FINAL EVALUATION ==========")
print("Questions tested:", len(scores))
print(f"Average Precision@3: {average_precision:.2%}")
