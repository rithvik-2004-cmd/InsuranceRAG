
from pathlib import Path
import os
import time
from datetime import datetime

import chromadb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Initialize ChromaDB only if the collection is missing or empty.
def ensure_database():
    db_dir = BASE_DIR / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_dir))

    try:
        collection = client.get_collection("insurance_policies")
        if collection.count() > 0:
            print("Existing ChromaDB collection found.")
            return
    except Exception:
        pass

    print("ChromaDB is missing or empty. Building it from insurance_docs...")
    from step9_ingest_documents import ingest_documents
    ingest_documents()


ensure_database()

# Import only after the database is ready.
import step12_chroma_rag_chat as rag

NO_INFO = (
    "The provided policy documents do not contain "
    "enough information to answer that."
)

rag.initialize_analytics()

def run_rag(question: str) -> dict:
    total_start = time.perf_counter()

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
        "embedding_time_seconds": 0,
        "retrieval_time_seconds": 0,
        "chat_time_seconds": 0,
        "total_time_seconds": 0,
        "retrieved_sections": 0,
        "answer_status": "error",
        "accuracy_status": "not_evaluated",
    }

    try:
        documents = rag.retrieve_documents(question, record)

        if not documents:
            answer = NO_INFO
            record["answer_status"] = "no_relevant_documents"
        else:
            answer, chat_metrics = rag.generate_answer(question, documents)
            record.update(chat_metrics)
            record["answer_status"] = "answered"

        return {
            "answer": answer,
            "sources": [
                {
                    "title": (
                        f"{d['source']} · Section "
                        f"{d['section_number']}: {d['section_title']}"
                    ),
                    "snippet": d["text"],
                }
                for d in documents
            ],
        }
    finally:
        record["total_time_seconds"] = round(
            time.perf_counter() - total_start, 2
        )
        rag.save_analytics(record)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    question: str
    session_id: str | None = None

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/chat")
def chat(body: ChatIn):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    try:
        return run_rag(question)
    except requests.RequestException as e:
        raise HTTPException(502, f"NVIDIA API request failed: {e}")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
