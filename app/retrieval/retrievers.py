from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from qdrant_client.models import Filter, FieldCondition, MatchAny, SparseVector, FusionQuery, Prefetch

from app.database import SessionLocal, BlueprintPage
from app.ingestion.colqwen2_embedder import embedder as colqwen2_embedder
from app.ingestion.bge_embedder import bge_embedder
from app.qdrant_setup import get_client


class BlueprintRetriever(BaseRetriever):
    document_ids: list = []
    top_k: int = 50

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        try:
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

            results = qdrant.search(
                collection_name="blueprint_visual",
                query_vector=("colqwen2", query_vecs[0] if query_vecs else []),
                query_filter=search_filter,
                limit=self.top_k,
            )

            db = SessionLocal()
            docs = []
            try:
                for hit in results:
                    point_id = hit.id
                    page = (
                        db.query(BlueprintPage)
                        .filter(BlueprintPage.qdrant_point_id == str(point_id))
                        .first()
                    )
                    text = page.numarkdown_text if page and page.numarkdown_text else "No text extracted"
                    docs.append(
                        LCDocument(
                            page_content=text,
                            metadata={
                                "document_id": hit.payload.get("document_id"),
                                "page_number": hit.payload.get("page_number"),
                                "image_path": hit.payload.get("image_path"),
                                "source": "blueprint",
                            },
                        )
                    )
            finally:
                db.close()

            return docs

        except Exception:
            return []


class TextRetriever(BaseRetriever):
    document_ids: list = []
    top_k: int = 50

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
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=self.top_k,
                        filter=search_filter,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=self.top_k,
            ).points

            docs = []
            for hit in results:
                docs.append(
                    LCDocument(
                        page_content=hit.payload.get("content", ""),
                        metadata={
                            "document_id": hit.payload.get("document_id"),
                            "chunk_index": hit.payload.get("chunk_index"),
                            "source": "spec",
                        },
                    )
                )

            return docs

        except Exception:
            return []
