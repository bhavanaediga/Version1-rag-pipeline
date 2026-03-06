import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv()

from app.database import get_db, Document, Base, engine
from app.qdrant_setup import initialize_collections
from app.ingestion.blueprint_pipeline import ingest_blueprint
from app.ingestion.text_pipeline import ingest_text_document
from app.agents.rag_graph import run_query

app = FastAPI(title="PlanSmartAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)
    initialize_collections()


UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    file_type: str = Form(...),
    db: Session = Depends(get_db),
):
    if file_type not in {"blueprint", "spec", "both"}:
        raise HTTPException(status_code=400, detail="file_type must be 'blueprint', 'spec', or 'both'")

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{file.filename}"
    file_path = str(UPLOAD_DIR / safe_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    doc = Document(
        id=doc_id,
        file_name=file.filename,
        file_type=file_type,
        file_path=file_path,
        processing_status="pending",
    )
    db.add(doc)
    db.commit()

    if file_type == "blueprint":
        background_tasks.add_task(ingest_blueprint, doc_id, file_path, next(get_db()))
    elif file_type == "spec":
        background_tasks.add_task(ingest_text_document, doc_id, file_path, next(get_db()))
    else:
        # "both" — run visual + text pipelines concurrently on the same document
        background_tasks.add_task(ingest_blueprint, doc_id, file_path, next(get_db()))
        background_tasks.add_task(ingest_text_document, doc_id, file_path, next(get_db()))

    return {"document_id": doc_id, "status": "processing"}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return [
        {
            "id": str(d.id),
            "file_name": d.file_name,
            "file_type": d.file_type,
            "processing_status": d.processing_status,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in docs
    ]


@app.get("/documents/{document_id}/status")
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "status": doc.processing_status,
        "progress_detail": doc.progress_detail or "",
    }


class QueryRequest(BaseModel):
    query: str
    document_ids: list[str] = []


@app.post("/query")
def query_documents(req: QueryRequest, db: Session = Depends(get_db)):
    result = run_query(req.query, req.document_ids)
    return {
        "answer": result["answer"],
        "cited_sources": result["cited_sources"],
        "confidence_score": result["confidence_score"],
    }
