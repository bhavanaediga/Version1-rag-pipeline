import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def initialize_collections() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if "blueprint_visual" not in existing:
        client.create_collection(
            collection_name="blueprint_visual",
            vectors_config={
                "colqwen2": VectorParams(size=128, distance=Distance.COSINE)
            },
        )
        print("Created collection: blueprint_visual")

    if "text_hybrid" not in existing:
        client.create_collection(
            collection_name="text_hybrid",
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )
        print("Created collection: text_hybrid")
