import re
import time
import uuid

from qdrant_client.models import PointStruct, SparseVector

from app.database import Document, TextChunk
from app.ingestion.bge_embedder import bge_embedder
from app.qdrant_setup import get_client


def extract_section_header(chunk: str) -> str:
    for line in chunk.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if (
            stripped == stripped.upper()
            and len(stripped) >= 10
            and re.search(r"[A-Z]", stripped)
            and not re.fullmatch(r"[\d\s\.\-/\\|]+", stripped)
        ):
            return stripped
    return ""


def update_progress(document_id: str, message: str, db) -> None:
    db.query(Document).filter(Document.id == document_id).update(
        {"progress_detail": message}, synchronize_session=False
    )
    db.commit()


async def ingest_text_document(document_id: str, file_path: str, db) -> None:
    doc = db.query(Document).filter(Document.id == document_id).first()
    doc.processing_status = "processing"
    db.commit()

    t_start = time.perf_counter()
    t_docling = t_split = t_embed = t_qdrant = t_pg = 0.0

    try:
        # ── Stage 1: Docling PDF → Markdown ──────────────────────────────────
        update_progress(document_id, "Parsing PDF with Docling", db)
        t0 = time.perf_counter()

        from docling.document_converter import DocumentConverter  # type: ignore
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.table_structure_options.do_cell_matching = False

        converter = DocumentConverter(pipeline_options=opts)
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        t_docling = time.perf_counter() - t0

        # ── Stage 2: Split into chunks ────────────────────────────────────────
        update_progress(document_id, "Splitting into chunks", db)
        t0 = time.perf_counter()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        )
        chunks = splitter.split_text(markdown_text)
        total = len(chunks)
        t_split = time.perf_counter() - t0

        # ── Stage 3: Batch BGE-M3 embedding (single call) ────────────────────
        update_progress(document_id, "Generating BGE-M3 embeddings", db)
        t0 = time.perf_counter()

        all_embeddings = bge_embedder.embed_texts(chunks)
        dense_vecs = all_embeddings["dense"]      # list of 1024-dim lists
        sparse_dicts = all_embeddings["sparse"]   # list of {token_id: weight}
        t_embed = time.perf_counter() - t0

        # ── Stage 4: Build points + upsert in batches of 50 ──────────────────
        update_progress(document_id, "Indexing into Qdrant", db)
        t0 = time.perf_counter()

        qdrant = get_client()
        BATCH_SIZE = 50
        all_db_records: list[TextChunk] = []
        all_points: list[PointStruct] = []
        batch_points: list[PointStruct] = []

        for i, chunk in enumerate(chunks):
            section_header = extract_section_header(chunk)
            sparse_dict = sparse_dicts[i]
            sparse_indices = [int(k) for k in sparse_dict.keys()]
            sparse_values = [float(v) for v in sparse_dict.values()]

            point_id = str(uuid.uuid4())
            batch_points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vecs[i],
                        "sparse": SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                    },
                    payload={
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": chunk,
                        "section_header": section_header,
                    },
                )
            )
            all_db_records.append(
                TextChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk,
                    section_header=section_header,
                    qdrant_point_id=point_id,
                )
            )

            if len(batch_points) >= BATCH_SIZE:
                qdrant.upsert(collection_name="text_hybrid", points=batch_points)
                batch_points = []

        if batch_points:
            qdrant.upsert(collection_name="text_hybrid", points=batch_points)

        t_qdrant = time.perf_counter() - t0

        # ── Stage 5: single bulk PostgreSQL insert ────────────────────────────
        update_progress(document_id, "Saving to PostgreSQL", db)
        t0 = time.perf_counter()
        db.bulk_save_objects(all_db_records)
        doc.processing_status = "done"
        update_progress(document_id, "Done", db)
        db.commit()
        t_pg = time.perf_counter() - t0

        # ── Timing report ─────────────────────────────────────────────────────
        t_total = time.perf_counter() - t_start
        print(
            f"\n[text_pipeline] Timing for {total} chunks:\n"
            f"  Docling parsing:    {t_docling:.1f}s\n"
            f"  Chunk splitting:    {t_split:.1f}s\n"
            f"  BGE-M3 embedding:   {t_embed:.1f}s\n"
            f"  Qdrant upsert:      {t_qdrant:.1f}s\n"
            f"  PostgreSQL insert:  {t_pg:.1f}s\n"
            f"  Total:              {t_total:.1f}s  "
            f"({t_total / total:.3f}s per chunk)\n"
        )

    except Exception as exc:
        doc.processing_status = "failed"
        update_progress(document_id, f"Failed: {exc}", db)
        db.commit()
        raise exc
