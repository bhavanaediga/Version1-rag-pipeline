import asyncio
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium  # type: ignore
from qdrant_client.models import PointStruct

from app.database import Document, BlueprintPage
from app.ingestion.colqwen2_embedder import embedder as colqwen2_embedder
from app.qdrant_setup import get_client


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_page_count(file_path: str) -> int:
    doc = pdfium.PdfDocument(file_path)
    n = len(doc)
    doc.close()
    return n


def _convert_page(args: tuple) -> tuple[int, str]:
    """Convert one PDF page to JPEG and return (page_number, image_path)."""
    file_path, image_dir, n = args
    from pdf2image import convert_from_path  # type: ignore
    pages = convert_from_path(
        file_path,
        dpi=250,
        first_page=n,
        last_page=n,
        use_pdftocairo=True,
        fmt="jpeg",
        jpegopt={"quality": 92},
    )
    image_path = str(Path(image_dir) / f"page_{n}.jpg")
    pages[0].save(image_path, "JPEG", quality=92)
    return n, image_path


def _extract_text(args: tuple) -> tuple[int, str]:
    """Extract text from one PDF page and return (page_number, text)."""
    file_path, n = args
    try:
        doc = pdfium.PdfDocument(file_path)
        page = doc[n - 1]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded().strip()
        doc.close()
        return n, text
    except Exception:
        return n, ""


def _run_conversions(conv_args: list) -> list:
    """Run all page conversions in a dedicated thread pool (no nesting)."""
    workers = min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_convert_page, conv_args, chunksize=4))


def _run_text_extractions(text_args: list) -> list:
    """Run all text extractions in a dedicated thread pool (no nesting)."""
    workers = min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_extract_text, text_args, chunksize=8))


def _run_embeddings(image_paths: list) -> list:
    """Run ColQwen2 batch embedding (CPU-intensive, run in executor)."""
    return colqwen2_embedder.embed_pages_batch(image_paths, batch_size=2)


def update_progress(document_id: str, message: str, db) -> None:
    db.query(Document).filter(Document.id == document_id).update(
        {"progress_detail": message}, synchronize_session=False
    )
    db.commit()


# ── main pipeline ─────────────────────────────────────────────────────────────

async def ingest_blueprint(document_id: str, file_path: str, db) -> None:
    doc = db.query(Document).filter(Document.id == document_id).first()
    doc.processing_status = "processing"
    db.commit()

    t_start = time.perf_counter()
    t_conv = t_text = t_embed = t_qdrant = t_pg = 0.0

    # Dedicated executor for blocking I/O (conversion + text extraction)
    io_executor = ThreadPoolExecutor(max_workers=2)
    # Dedicated executor for CPU-bound embedding (single thread to avoid contention)
    cpu_executor = ThreadPoolExecutor(max_workers=1)

    try:
        total = _get_page_count(file_path)
        image_dir = Path(f"storage/images/{document_id}")
        image_dir.mkdir(parents=True, exist_ok=True)

        image_dir_str = str(image_dir)
        conv_args = [(file_path, image_dir_str, n) for n in range(1, total + 1)]
        text_args = [(file_path, n) for n in range(1, total + 1)]

        loop = asyncio.get_event_loop()

        # ── Stage 1 & 2: PDF→images and text extraction run concurrently ─────
        update_progress(document_id, f"Converting {total} pages to images + extracting text", db)

        t0 = time.perf_counter()
        conv_future = loop.run_in_executor(io_executor, _run_conversions, conv_args)
        text_future = loop.run_in_executor(io_executor, _run_text_extractions, text_args)
        conv_results, text_results = await asyncio.gather(conv_future, text_future)
        t_conv = time.perf_counter() - t0
        t_text = t_conv  # concurrent

        image_map: dict[int, str] = {n: path for n, path in conv_results}
        text_map: dict[int, str] = {n: text for n, text in text_results}

        # ── Stage 3: ColQwen2 batch embedding (offloaded to cpu_executor) ─────
        update_progress(document_id, f"Generating ColQwen2 embeddings for {total} pages", db)
        t0 = time.perf_counter()

        image_paths_ordered = [image_map[n] for n in range(1, total + 1)]
        all_embeddings = await loop.run_in_executor(
            cpu_executor, _run_embeddings, image_paths_ordered
        )
        t_embed = time.perf_counter() - t0

        # ── Stage 4: Qdrant upsert in batches of 5 ───────────────────────────
        update_progress(document_id, "Indexing into Qdrant", db)
        t0 = time.perf_counter()

        qdrant = get_client()
        BATCH_SIZE = 5
        all_db_records: list[BlueprintPage] = []
        batch_points: list[PointStruct] = []

        for idx, n in enumerate(range(1, total + 1)):
            point_id = str(uuid.uuid4())
            batch_points.append(
                PointStruct(
                    id=point_id,
                    vector={"colqwen2": all_embeddings[idx]},
                    payload={
                        "document_id": document_id,
                        "page_number": n,
                        "image_path": image_map[n],
                    },
                )
            )
            all_db_records.append(
                BlueprintPage(
                    document_id=document_id,
                    page_number=n,
                    image_path=image_map[n],
                    extracted_text=text_map[n],
                    qdrant_point_id=point_id,
                )
            )
            if len(batch_points) >= BATCH_SIZE:
                qdrant.upsert(collection_name="blueprint_visual", points=batch_points)
                batch_points = []

        if batch_points:
            qdrant.upsert(collection_name="blueprint_visual", points=batch_points)

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
            f"\n[blueprint_pipeline] Timing for {total} pages:\n"
            f"  PDF conversion + text: {t_conv:.1f}s  (concurrent)\n"
            f"  ColQwen2 embedding:    {t_embed:.1f}s\n"
            f"  Qdrant upsert:         {t_qdrant:.1f}s\n"
            f"  PostgreSQL insert:     {t_pg:.1f}s\n"
            f"  Total:                 {t_total:.1f}s  "
            f"({t_total / total:.2f}s per page)\n"
        )

    except Exception as exc:
        doc.processing_status = "failed"
        update_progress(document_id, f"Failed: {exc}", db)
        db.commit()
        raise exc

    finally:
        io_executor.shutdown(wait=False)
        cpu_executor.shutdown(wait=False)
