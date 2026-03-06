import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    SparseVectorParams,
    VectorParams,
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

    # ── blueprint_visual ──────────────────────────────────────────────────────
    if "blueprint_visual" in existing:
        info = client.get_collection("blueprint_visual")
        vectors_cfg = info.config.params.vectors
        colqwen2_cfg = (
            vectors_cfg.get("colqwen2") if isinstance(vectors_cfg, dict) else None
        )
        needs_recreate = (
            colqwen2_cfg is None or colqwen2_cfg.multivector_config is None
        )
        if needs_recreate:
            print(
                "Old blueprint_visual schema detected — recreating with MultiVectorConfig"
            )
            client.delete_collection("blueprint_visual")
            existing.discard("blueprint_visual")

    if "blueprint_visual" not in existing:
        client.create_collection(
            collection_name="blueprint_visual",
            vectors_config={
                "colqwen2": VectorParams(
                    size=128,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )
        print("Created collection: blueprint_visual (multi-vector MAX_SIM)")

    # ── text_hybrid ───────────────────────────────────────────────────────────
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
        print("Created collection: text_hybrid (dense + sparse RRF)")
