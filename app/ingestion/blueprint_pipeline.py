import os
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from app.database import Document, BlueprintPage
from app.ingestion.colqwen2_embedder import embedder as colqwen2_embedder
from app.ingestion.numarkdown import extract_text_from_image
from app.qdrant_setup import get_client


async def ingest_blueprint(document_id: str, file_path: str, db) -> None:
    doc = db.query(Document).filter(Document.id == document_id).first()
    doc.processing_status = "processing"
    db.commit()

    try:
        from pdf2image import convert_from_path  # type: ignore

        pages = convert_from_path(file_path, dpi=200)
        total = len(pages)

        image_dir = Path(f"storage/images/{document_id}")
        image_dir.mkdir(parents=True, exist_ok=True)

        qdrant = get_client()
        points = []
        db_records = []

        for n, page_img in enumerate(pages, start=1):
            print(f"Processing page {n} of {total}...")
            image_path = str(image_dir / f"page_{n}.png")
            page_img.save(image_path, "PNG")

            multi_vecs = colqwen2_embedder.embed_page_image(image_path)
            numarkdown_text = extract_text_from_image(image_path)

            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={"colqwen2": multi_vecs[0] if multi_vecs else []},
                    payload={
                        "document_id": document_id,
                        "page_number": n,
                        "image_path": image_path,
                    },
                )
            )

            db_records.append(
                BlueprintPage(
                    document_id=document_id,
                    page_number=n,
                    image_path=image_path,
                    numarkdown_text=numarkdown_text,
                    qdrant_point_id=point_id,
                )
            )

        qdrant.upsert(collection_name="blueprint_visual", points=points)
        db.add_all(db_records)

        doc.processing_status = "done"
        db.commit()

    except Exception as exc:
        doc.processing_status = "failed"
        db.commit()
        raise exc
