from __future__ import annotations

import torch
from PIL import Image


class ColQwen2Embedder:
    def __init__(self):
        from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    def embed_page_image(self, image_path: str) -> list[list[float]]:
        image = Image.open(image_path).convert("RGB")
        batch = self.processor.process_images([image]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)
        return embeddings[0].cpu().float().tolist()

    def embed_query(self, query_text: str) -> list[list[float]]:
        batch = self.processor.process_queries([query_text]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)
        return embeddings[0].cpu().float().tolist()

    def compute_max_sim(
        self, query_vecs: list[list[float]], doc_vecs: list[list[float]]
    ) -> float:
        import torch.nn.functional as F

        q = torch.tensor(query_vecs, dtype=torch.float32)
        d = torch.tensor(doc_vecs, dtype=torch.float32)
        q_norm = F.normalize(q, dim=-1)
        d_norm = F.normalize(d, dim=-1)
        sim_matrix = torch.matmul(q_norm, d_norm.T)
        max_sims = sim_matrix.max(dim=1).values
        return float(max_sims.sum().item())


_embedder: ColQwen2Embedder | None = None


def get_embedder() -> ColQwen2Embedder:
    global _embedder
    if _embedder is None:
        _embedder = ColQwen2Embedder()
    return _embedder


# Backward-compatible lazy alias
class _LazyEmbedder:
    def embed_page_image(self, image_path):
        return get_embedder().embed_page_image(image_path)

    def embed_query(self, query_text):
        return get_embedder().embed_query(query_text)

    def compute_max_sim(self, query_vecs, doc_vecs):
        return get_embedder().compute_max_sim(query_vecs, doc_vecs)


embedder = _LazyEmbedder()
