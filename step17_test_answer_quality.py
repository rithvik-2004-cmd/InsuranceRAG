
import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv

# Project setup
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("NVIDIA_API_KEY")

EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
CHAT_MODEL = "z-ai/glm-5.3-flash"

# Connect to the existing database
client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)
collection = client.get_collection(name="insurance_policies")

THRESHOLDS = [1.0, 1.2, 1.5]

TEST_QUESTIONS = [
    "What is a deductible?",
    "What expenses are excluded from health insurance?",
    "What should I do if my insured car is stolen?",
    "What is the capital of Japan?",
]


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
            "encoding_format": "float",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def retrieve_documents(question, threshold, top_k=3):
    embedding = get_query_embedding(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance <= threshold:
            documents.append({
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "section_title": metadata.get(
                    "section_title", "Unknown"
                ),
                "distance": distance,
            })

    return documents


def generate_answer(question, documents):
    if not documents:
        return (
            "No documents passed the distance threshold. "
            "The chatbot should not answer from retrieved context."
        )

    context = "\n\n---\n\n".join(
        f"Source: {doc['source']}\n"
        f"Section: {doc['section_title']}\n"
        f"Content:\n{doc['text']}"
        for doc in documents
    )

    prompt = f"""
You are an insurance assistant for a learning project.

Answer using only the supplied insurance context.
Do not invent policy terms, amounts, deadlines, or requirements.
If the context does not contain enough information, say:
"The provided policy documents do not contain enough information to answer that."
If the question is unrelated to insurance, politely explain that
you specialize in insurance-related questions.
Use simple language and mention source filenames.
These are fictional learning documents.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = requests.post(
        CHAT_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer insurance questions accurately "
                        "using only the supplied context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": False,
        },
        timeout=90,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing from .env")

    print("Documents stored:", collection.count())
    print("Starting answer-quality comparison...")

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 75)
        print("QUESTION:", question)

        for threshold in THRESHOLDS:
            print("\n" + "-" * 60)
            print("DISTANCE THRESHOLD:", threshold)

            documents = retrieve_documents(question, threshold)

            print("Sections retrieved:", len(documents))
            for doc in documents:
                print(
                    f"- {doc['source']} | {doc['section_title']} "
                    f"| distance={doc['distance']:.3f}"
                )

            print("\nGENERATED ANSWER:")
            answer = generate_answer(question, documents)
            print(answer)
