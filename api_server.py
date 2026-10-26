"""
api_server.py  -  put next to step12_chroma_rag_chat.py (same folder as .env and chroma_db)
Install:  pip install fastapi uvicorn
Run:      uvicorn api_server:app --reload --port 8000
Open:     http://localhost:8000
"""
import time
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import step12_chroma_rag_chat as rag

NO_INFO = ("The provided policy documents do not contain "
           "enough information to answer that.")

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

    documents = rag.retrieve_documents(question, record)

    if not documents:
        answer = NO_INFO
        record["answer_status"] = "no_relevant_documents"
    else:
        answer, chat_metrics = rag.generate_answer(question, documents)
        record.update(chat_metrics)
        record["answer_status"] = "answered"

    record["total_time_seconds"] = round(time.perf_counter() - total_start, 2)
    rag.save_analytics(record)

    sources = [
        {
            "title": f"{d['source']} · Section {d['section_number']}: {d['section_title']}",
            "snippet": d["text"],
        }
        for d in documents
    ]
    return {"answer": answer, "sources": sources}


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")