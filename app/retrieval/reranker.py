from langchain_core.documents import Document as LCDocument

from app.ingestion.bge_embedder import reranker as _reranker

TEXT_THRESHOLD = 30  # chars — below this a page is treated as visual-only


def rerank_documents(
    query: str, documents: list[LCDocument], top_k: int = 10
) -> list[LCDocument]:
    if not documents:
        return []

    # ── Group 1: pinned (directly injected by page number) ────────────────────
    pinned = [d for d in documents if d.metadata.get("pinned")]
    unpinned = [d for d in documents if not d.metadata.get("pinned")]
    for doc in pinned:
        doc.metadata["reranker_score"] = 10.0

    # ── Group 2 & 3: split unpinned by text richness ──────────────────────────
    text_rich = [d for d in unpinned if len(d.page_content) >= TEXT_THRESHOLD]
    visual_only = [d for d in unpinned if len(d.page_content) < TEXT_THRESHOLD]

    # Group 2: BGE cross-encoder on text-rich docs (truncated to 512 chars)
    reranked_text: list[LCDocument] = []
    if text_rich:
        passages = [doc.page_content[:512] for doc in text_rich]
        ranked = _reranker.rerank(query, passages)
        for orig_idx, score in ranked:
            doc = text_rich[orig_idx]
            doc.metadata["reranker_score"] = float(score)
            reranked_text.append(doc)

    # Group 3: visual-only — sort by ColQwen2 MaxSim, map to [-10, 10]
    # ColQwen2 MaxSim (summed cosine over query tokens) is typically in [0, ~20].
    # Map: reranker_score = clamp(raw - 10, -10, 10)
    visual_sorted = sorted(
        visual_only,
        key=lambda d: d.metadata.get("colqwen2_score", 0.0),
        reverse=True,
    )
    for doc in visual_sorted:
        raw = doc.metadata.get("colqwen2_score", 0.0)
        doc.metadata["reranker_score"] = max(-10.0, min(10.0, raw - 10.0))

    # ── Merge: pinned first, then rest sorted by score descending ─────────────
    rest = sorted(
        reranked_text + visual_sorted,
        key=lambda d: d.metadata.get("reranker_score", -10.0),
        reverse=True,
    )
    return (pinned + rest)[:top_k]


def compute_confidence(docs: list[LCDocument]) -> float:
    scores = [
        float(d.metadata["reranker_score"])
        for d in docs
        if d.metadata.get("reranker_score") is not None
    ]
    if not scores:
        return 0.0
    top3_avg = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
    # Scores are in [-10, 10]; normalise to [0, 1]
    return round(max(0.0, min(1.0, (top3_avg + 10) / 20)), 4)
