import uuid

from qdrant_client.models import PointStruct, SparseVector

from app.database import Document, TextChunk
from app.ingestion.bge_embedder import bge_embedder
from app.qdrant_setup import get_client


async def ingest_text_document(document_id: str, file_path: str, db) -> None:
    doc = db.query(Document).filter(Document.id == document_id).first()
    doc.processing_status = "processing"
    db.commit()

    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(markdown_text)
        total = len(chunks)

        qdrant = get_client()
        points = []
        db_records = []

        for i, chunk in enumerate(chunks):
            print(f"Indexing chunk {i + 1} of {total}...")
            embedding = bge_embedder.embed_single(chunk)

            dense_vec = embedding["dense"]
            sparse_dict = embedding["sparse"]

            sparse_indices = [int(k) for k in sparse_dict.keys()]
            sparse_values = [float(v) for v in sparse_dict.values()]

            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                    },
                    payload={
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": chunk,
                    },
                )
            )

            db_records.append(
                TextChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk,
                    qdrant_point_id=point_id,
                )
            )

        qdrant.upsert(collection_name="text_hybrid", points=points)
        db.add_all(db_records)

        doc.processing_status = "done"
        db.commit()

    except Exception as exc:
        doc.processing_status = "failed"
        db.commit()
        raise exc
