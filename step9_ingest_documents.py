import os
import re
import hashlib
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv


# Project configuration
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "insurance_docs"
DB_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

if not API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is missing from your .env file.")


def split_into_sections(text):
    """Split a document and extract each section's heading."""

    parts = re.split(r"(?m)(?=^\d+\.\s+)", text.strip())
    sections = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        first_line = part.splitlines()[0].strip()
        match = re.match(r"^(\d+)\.\s+(.+)$", first_line)

        if match:
            section_number = int(match.group(1))
            section_title = match.group(2).strip()
        else:
            section_number = 0
            section_title = "Document Information"

        sections.append({
            "text": part,
            "section_number": section_number,
            "section_title": section_title,
        })

    return sections


def get_embeddings(texts):
    """Generate embeddings for a batch of text sections."""

    response = requests.post(
        EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
            "input_type": "passage",
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])

    return [item["embedding"] for item in data]


def extract_document_type(text):
    """Extract the document type from the source text."""

    match = re.search(r"(?m)^DOCUMENT TYPE:\s*(.+)$", text)

    return match.group(1).strip() if match else "General"


def ingest_documents():
    """Read documents, create embeddings, and store all sections."""

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="insurance_policies",
        metadata={"hnsw:space": "l2"},
    )

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCS_DIR}"
        )

    files = sorted(DOCS_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {DOCS_DIR}"
        )

    # IMPORTANT: Initialize once, before processing any files.
    all_records = []

    # Process every document.
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        sections = split_into_sections(text)
        document_type = extract_document_type(text)

        print(f"\nProcessing: {file_path.name}")
        print(f"Sections found: {len(sections)}")

        # Process every section in the current document.
        for section in sections:
            section_number = section["section_number"]
            section_title = section["section_title"]
            section_text = section["text"]

            # Stable ID based on filename, section number and title.
            record_id = hashlib.sha256(
                (
                    f"{file_path.name}:"
                    f"{section_number}:"
                    f"{section_title}"
                ).encode("utf-8")
            ).hexdigest()

            all_records.append({
                "id": record_id,
                "text": section_text,
                "metadata": {
                    "source": file_path.name,
                    "document_type": document_type,
                    "section_number": section_number,
                    "section_title": section_title,
                },
            })

    print(f"\nTotal sections to ingest: {len(all_records)}")

    # Generate embeddings and store records in batches.
    batch_size = 20

    for start in range(0, len(all_records), batch_size):
        batch = all_records[start:start + batch_size]

        embeddings = get_embeddings([
            record["text"] for record in batch
        ])

        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=embeddings,
            metadatas=[
                record["metadata"] for record in batch
            ],
        )

        print(
            f"Stored {min(start + batch_size, len(all_records))}"
            f"/{len(all_records)} sections"
        )

    print("\nIngestion complete!")
    print("Total sections processed:", len(all_records))
    print("Total records currently in collection:", collection.count())
    print("=" * 60)
    print("Collection :", collection.id)
   

if __name__ == "__main__":
    ingest_documents()
