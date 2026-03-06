import re
import uuid as _uuid
from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchAny,
    Prefetch,
    SparseVector,
    Fusion,
)

from app.database import SessionLocal, BlueprintPage
from app.ingestion.colqwen2_embedder import embedder as colqwen2_embedder
from app.ingestion.bge_embedder import bge_embedder
from app.qdrant_setup import get_client


def _extract_page_number(query: str) -> int | None:
    """Return the first explicit page number mentioned in query, or None."""
    m = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
    return int(m.group(1)) if m else None


class BlueprintRetriever(BaseRetriever):
    document_ids: list = []
    top_k: int = 20  # ColQwen2 MaxSim ranking is high quality; 20 is enough

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        try:
            db = SessionLocal()
            try:
                # ── OPT-9: page-number injection — skip Qdrant entirely ──────
                page_num = _extract_page_number(query)
                if page_num is not None and self.document_ids:
                    doc_uuids = [_uuid.UUID(did) for did in self.document_ids]
                    page = (
                        db.query(BlueprintPage)
                        .filter(
                            BlueprintPage.page_number == page_num,
                            BlueprintPage.document_id.in_(doc_uuids),
                        )
                        .first()
                    )
                    if page:
                        return [
                            LCDocument(
                                page_content=page.extracted_text or "",
                                metadata={
                                    "document_id": str(page.document_id),
                                    "page_number": page.page_number,
                                    "image_path": page.image_path,
                                    "source": "blueprint",
                                    "colqwen2_score": 999.0,
                                    "pinned": True,
                                },
                            )
                        ]

                # ── ColQwen2 MaxSim vector search ─────────────────────────────
                query_vecs = colqwen2_embedder.embed_query(query)
                qdrant = get_client()

                search_filter = None
                if self.document_ids:
                    search_filter = Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchAny(any=self.document_ids),
                            )
                        ]
                    )

                results = qdrant.query_points(
                    collection_name="blueprint_visual",
                    query=query_vecs,
                    using="colqwen2",
                    query_filter=search_filter,
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                ).points

                docs: List[LCDocument] = []
                for hit in results:
                    page = (
                        db.query(BlueprintPage)
                        .filter(BlueprintPage.qdrant_point_id == str(hit.id))
                        .first()
                    )
                    text = page.extracted_text if page and page.extracted_text else ""
                    docs.append(
                        LCDocument(
                            page_content=text,
                            metadata={
                                "document_id": hit.payload.get("document_id"),
                                "page_number": hit.payload.get("page_number"),
                                "image_path": hit.payload.get("image_path"),
                                "source": "blueprint",
                                "colqwen2_score": float(hit.score),
                            },
                        )
                    )
                return docs

            finally:
                db.close()

        except Exception:
            return []


class TextRetriever(BaseRetriever):
    document_ids: list = []
    top_k: int = 100

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        try:
            embedding = bge_embedder.embed_single(query)
            dense_vec = embedding["dense"]
            sparse_dict = embedding["sparse"]
            sparse_indices = [int(k) for k in sparse_dict.keys()]
            sparse_values = [float(v) for v in sparse_dict.values()]

            qdrant = get_client()

            search_filter = None
            if self.document_ids:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchAny(any=self.document_ids),
                        )
                    ]
                )

            # ── OPT-8: proper RRF fusion query ────────────────────────────────
            results = qdrant.query_points(
                collection_name="text_hybrid",
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=self.top_k,
                        filter=search_filter,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                        using="sparse",
                        limit=self.top_k,
                        filter=search_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=self.top_k,
                with_payload=True,
            ).points

            docs: List[LCDocument] = []
            for hit in results:
                docs.append(
                    LCDocument(
                        page_content=hit.payload.get("content", ""),
                        metadata={
                            "document_id": hit.payload.get("document_id"),
                            "chunk_index": hit.payload.get("chunk_index"),
                            "section_header": hit.payload.get("section_header", ""),
                            "source": "spec",
                        },
                    )
                )
            return docs

        except Exception:
            return []
