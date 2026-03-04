from langchain_core.documents import Document as LCDocument

from app.ingestion.bge_embedder import reranker as _reranker


def rerank_documents(
    query: str, documents: list[LCDocument], top_k: int = 10
) -> list[LCDocument]:
    if not documents:
        return []

    passages = [doc.page_content for doc in documents]
    ranked = _reranker.rerank(query, passages)

    reranked = []
    for orig_idx, score in ranked[:top_k]:
        doc = documents[orig_idx]
        doc.metadata["reranker_score"] = score
        reranked.append(doc)

    return reranked


def compute_confidence(docs: list[LCDocument]) -> float:
    scores = []
    for doc in docs:
        score = doc.metadata.get("reranker_score")
        if score is not None:
            scores.append(float(score))

    if not scores:
        return 0.0

    top_scores = sorted(scores, reverse=True)[:3]
    avg = sum(top_scores) / len(top_scores)

    # Normalize: reranker scores are logits, typically in range [-10, 10]
    normalized = max(0.0, min(1.0, (avg + 10) / 20))
    return round(normalized, 4)
