
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


# 3. Connect to existing ChromaDB
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_collection(
    name="insurance_policies"
)

print("Environment loaded:", bool(API_KEY))
print("Documents stored:", collection.count())


# 4. Generate an embedding for a question
def get_query_embedding(question):
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


# 5. Test questions and relevant section titles
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
            "Claims Process",
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


# 6. Test different distance thresholds
top_k = 3
thresholds = [1.0, 1.2, 1.5]

for threshold in thresholds:
    question_precisions = []
    total_retrieved = 0
    total_relevant = 0

    print(f"\n{'=' * 55}")
    print(f"Testing threshold: {threshold}")
    print(f"Retrieving up to {top_k} sections per question")
    print("=" * 55)

    for test in evaluation_questions:
        question = test["question"]
        relevant_sections = test["relevant_sections"]

        query_embedding = get_query_embedding(question)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )

        metadata_list = results["metadatas"][0]
        distance_list = results["distances"][0]

        # Keep only sections within the threshold
        filtered_results = [
            (metadata, distance)
            for metadata, distance in zip(
                metadata_list, distance_list
            )
            if distance <= threshold
        ]

        relevant_count = sum(
            metadata.get("section_title", "Unknown")
            in relevant_sections
            for metadata, distance in filtered_results
        )

        retrieved_count = len(filtered_results)

        # Precision among the sections that passed the threshold
        precision = (
            relevant_count / retrieved_count
            if retrieved_count
            else 0
        )

        question_precisions.append(precision)
        total_retrieved += retrieved_count
        total_relevant += relevant_count

        print(
            f"\n{question}\n"
            f"  Retrieved: {retrieved_count}"
            f" | Relevant: {relevant_count}"
            f" | Precision: {precision:.2%}"
        )

    average_precision = (
        sum(question_precisions) / len(question_precisions)
        if question_precisions
        else 0
    )

    print(f"\nAverage Precision: {average_precision:.2%}")
    print(f"Total sections retrieved: {total_retrieved}")
    print(f"Total relevant sections: {total_relevant}")
