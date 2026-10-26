
import os
import re
import hashlib
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "insurance_docs"
DB_DIR = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

BATCH_SIZE = 20


# --------------------------------------------------
# 2. Split documents into sections
# --------------------------------------------------

def split_into_sections(text):
    """Split numbered sections and 'Section:' headings."""

    pattern = r"(?m)(?=^(?:\d+\.\s+|Section:\s+))"
    parts = re.split(pattern, text.strip())
    sections = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        first_line = part.splitlines()[0].strip()

        numbered = re.match(r"^(\d+)\.\s+(.+)$", first_line)
        named = re.match(r"^Section:\s*(.+)$", first_line)

        if numbered:
            section_number = int(numbered.group(1))
            section_title = numbered.group(2).strip()

        elif named:
            section_number = 0
            section_title = named.group(1).strip()

        else:
            section_number = 0
            section_title = "Document Information"

        sections.append({
            "text": part,
            "section_number": section_number,
            "section_title": section_title,
        })

    return sections


# --------------------------------------------------
# 3. Generate embeddings using NVIDIA
# --------------------------------------------------

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

    embeddings = [item["embedding"] for item in data]

    if len(embeddings) != len(texts):
        raise ValueError(
            "Embedding count does not match input section count."
        )

    return embeddings


# --------------------------------------------------
# 4. Extract document type
# --------------------------------------------------

def extract_document_type(text):
    """Extract the document type from the source document."""

    match = re.search(r"(?m)^DOCUMENT TYPE:\s*(.+)$", text)

    if match:
        return match.group(1).strip()

    title_match = re.search(r"(?m)^DOCUMENT:\s*(.+)$", text)

    if title_match:
        return title_match.group(1).strip()

    title_match = re.search(r"(?m)^Document Title:\s*(.+)$", text)

    if title_match:
        return title_match.group(1).strip()

    return "General"

# --------------------------------------------------
# 5. Read and validate all documents
# --------------------------------------------------

def prepare_records():
    """Prepare records before changing the existing database."""

    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCS_DIR}"
        )

    files = sorted(DOCS_DIR.glob("*.txt"))
    
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {DOCS_DIR}"
        )

    all_records = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(
                f"Document is empty: {file_path.name}"
            )

        sections = split_into_sections(text)
        document_type = extract_document_type(text)

        print(f"\nProcessing: {file_path.name}")
        print(f"Sections detected: {len(sections)}")

        for section in sections:
            section_number = section["section_number"]
            section_title = section["section_title"]
            section_text = section["text"]

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

    if not all_records:
        raise ValueError("No sections were extracted.")

    print("\nDocuments found:", len(files))
    print("Total sections prepared:", len(all_records))

    return all_records


# --------------------------------------------------
# 6. Rebuild ChromaDB and ingest documents
# --------------------------------------------------

def ingest_documents():
    """Rebuild the collection and store updated sections."""

    if not API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing from your .env file."
        )

    # Validate and prepare all source records first.
    all_records = prepare_records()

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Remove the old collection to prevent stale records.
    existing_names = [
        collection.name for collection in client.list_collections()
    ]

    if "insurance_policies" in existing_names:
        print("\nDeleting existing insurance_policies collection...")
        client.delete_collection(name="insurance_policies")

    # Create a fresh collection.
    collection = client.create_collection(
        name="insurance_policies",
        metadata={"hnsw:space": "l2"},
    )

    print("\nCreated a fresh ChromaDB collection.")
    print("Starting embedding and storage...")

    # Embed and store sections in batches.
    for start in range(0, len(all_records), BATCH_SIZE):
        batch = all_records[start:start + BATCH_SIZE]

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

        completed = min(start + BATCH_SIZE, len(all_records))

        print(f"Stored {completed}/{len(all_records)} sections")

    # Final verification.
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print("Sections prepared:", len(all_records))
    print("Records stored:", collection.count())
    print("Collection:", collection.name)
    print("Database folder:", DB_DIR)


if __name__ == "__main__":
    ingest_documents()
