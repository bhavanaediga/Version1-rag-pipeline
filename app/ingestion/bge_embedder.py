from __future__ import annotations


class BGEM3Embedder:
    def __init__(self):
        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_texts(self, texts: list[str]) -> dict:
        output = self.model.encode(
            texts,
            batch_size=12,
            max_length=512,
            return_dense=True,
            return_sparse=True,
        )
        dense = output["dense_vecs"].tolist()
        sparse_raw = output["lexical_weights"]
        sparse = [
            {str(k): float(v) for k, v in s.items()} for s in sparse_raw
        ]
        return {"dense": dense, "sparse": sparse}

    def embed_single(self, text: str) -> dict:
        result = self.embed_texts([text])
        return {"dense": result["dense"][0], "sparse": result["sparse"][0]}


class BGEReranker:
    def __init__(self):
        from FlagEmbedding import FlagReranker  # type: ignore

        self.model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        pairs = [[query, p] for p in passages]
        scores = self.model.compute_score(pairs)
        if not isinstance(scores, list):
            scores = [scores]
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in indexed]


_bge_embedder: BGEM3Embedder | None = None
_reranker: BGEReranker | None = None


def get_bge_embedder() -> BGEM3Embedder:
    global _bge_embedder
    if _bge_embedder is None:
        _bge_embedder = BGEM3Embedder()
    return _bge_embedder


def get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker


# Backward-compatible aliases (lazy)
class _LazyBGEEmbedder:
    def embed_texts(self, texts):
        return get_bge_embedder().embed_texts(texts)

    def embed_single(self, text):
        return get_bge_embedder().embed_single(text)


class _LazyReranker:
    def rerank(self, query, passages):
        return get_reranker().rerank(query, passages)


bge_embedder = _LazyBGEEmbedder()
reranker = _LazyReranker()
