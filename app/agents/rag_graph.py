import base64
import io
import json
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from PIL import Image as PILImage

from app.retrieval.retrievers import BlueprintRetriever, TextRetriever
from app.retrieval.reranker import rerank_documents, compute_confidence

load_dotenv()

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="anthropic/claude-3.5-sonnet",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            temperature=0,
        )
    return _llm


class RAGState(TypedDict):
    query: str
    document_ids: list[str]
    query_type: str
    retrieved_docs: Annotated[list, operator.add]
    reranked_docs: list
    confidence: float
    answer: str
    cited_sources: list[dict]
    retry_count: int
    rewritten_query: str | None


# ── nodes ─────────────────────────────────────────────────────────────────────

def router_node(state: RAGState) -> dict:
    # If any queried document was ingested as "both", always run both retrievers
    if state["document_ids"]:
        from app.database import SessionLocal, Document as DBDocument
        import uuid as _uuid
        db = SessionLocal()
        try:
            doc_uuids = [_uuid.UUID(did) for did in state["document_ids"]]
            file_types = {
                d.file_type
                for d in db.query(DBDocument).filter(DBDocument.id.in_(doc_uuids)).all()
            }
        finally:
            db.close()
        if "both" in file_types:
            return {"query_type": "cross_document"}

    prompt = (
        "You are routing a construction document query to the correct retrieval system.\n\n"
        "Rules:\n"
        "- 'blueprint': ONLY when the query is explicitly about a specific page/sheet number, "
        "a named drawing (floor plan, elevation, section), or purely visual elements "
        "like layout, dimensions on a drawing, or symbols. "
        "Examples: 'what is on page 5', 'show me the reflected ceiling plan', "
        "'where is the fire alarm panel on the drawing'.\n"
        "- 'spec': ONLY when the query is clearly about written text content that would appear "
        "in specification documents — material specs, code citations, "
        "abbreviation definitions, written schedules.\n"
        "- 'cross_document': use this for ALL other queries, especially anything about "
        "requirements, systems, equipment, or topics that could appear in either drawings "
        "or text. When in doubt, use cross_document.\n\n"
        f"Query: {state['query']}\n\n"
        "Reply with exactly one word: blueprint, spec, or cross_document"
    )
    response = get_llm().invoke(prompt)
    query_type = response.content.strip().lower()
    if query_type not in {"blueprint", "spec", "cross_document"}:
        query_type = "cross_document"
    return {"query_type": query_type}


def blueprint_retrieve_node(state: RAGState) -> dict:
    q = state.get("rewritten_query") or state["query"]
    retriever = BlueprintRetriever(document_ids=state["document_ids"])
    docs = retriever.invoke(q)
    return {"retrieved_docs": docs}


def text_retrieve_node(state: RAGState) -> dict:
    q = state.get("rewritten_query") or state["query"]
    retriever = TextRetriever(document_ids=state["document_ids"])
    docs = retriever.invoke(q)
    return {"retrieved_docs": docs}


def rerank_node(state: RAGState) -> dict:
    q = state.get("rewritten_query") or state["query"]
    reranked = rerank_documents(q, state["retrieved_docs"])
    confidence = compute_confidence(reranked)
    return {"reranked_docs": reranked, "confidence": confidence}


def confidence_check_node(state: RAGState) -> str:
    # OPT-11: zero-result widen before confidence check
    if (
        len(state["retrieved_docs"]) == 0
        and state["query_type"] != "cross_document"
        and state["retry_count"] < 2
    ):
        return "widen"
    if state["confidence"] >= 0.4 or state["retry_count"] >= 2:
        return "generate"
    return "rewrite"


def widen_query_node(state: RAGState) -> dict:
    """Switch to cross_document when retrieval returned zero results."""
    return {
        "query_type": "cross_document",
        "retrieved_docs": [],
        "retry_count": state["retry_count"] + 1,
    }


def rewrite_query_node(state: RAGState) -> dict:
    prompt = (
        "The following query returned low-confidence results from a construction document search. "
        "Rewrite it to be more specific and detailed.\n\n"
        f"Original: {state['query']}\n\n"
        "Return only the rewritten query, nothing else."
    )
    response = get_llm().invoke(prompt)
    return {
        "rewritten_query": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }


# 3×3 grid tile labels — row/col names for clear spatial referencing
_TILE_LABELS = [
    (f"{row}-{col}", c / 3, r / 3, (c + 1) / 3, (r + 1) / 3)
    for r, row in enumerate(["top", "middle", "bottom"])
    for c, col in enumerate(["left", "center", "right"])
]


def _page_quadrants(image_path: str, page_num) -> list[dict]:
    """Split one blueprint page into a 3×3 grid (9 tiles), each upscaled to 2048px wide.
    Finer tiling captures small dimension callouts and annotation text more accurately.
    Returns a flat list of interleaved text+image_url dicts ready for the vision payload."""
    parts = []
    try:
        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        for label, xl, yl, xr, yr in _TILE_LABELS:
            crop = img.crop((int(xl * w), int(yl * h), int(xr * w), int(yr * h)))
            # upscale each tile to 2048px wide so fine text and dimensions are legible
            max_w = 2048
            scale = max_w / crop.width
            crop = crop.resize((max_w, int(crop.height * scale)), PILImage.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=92)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append({
                "type": "text",
                "text": f"[Blueprint page {page_num} — {label} tile]:",
            })
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    except Exception:
        parts.append({
            "type": "text",
            "text": f"[Blueprint page {page_num}]\n[Image not available]",
        })
    return parts


def generate_node(state: RAGState) -> dict:
    """Send spec docs as text + blueprint pages as 3×3 tile crops for fine spatial detail."""
    blueprint_docs = [
        d for d in state["reranked_docs"] if d.metadata.get("source") == "blueprint"
    ]
    spec_docs = [
        d for d in state["reranked_docs"] if d.metadata.get("source") != "blueprint"
    ]

    # Build text context from spec chunks
    text_parts = []
    for doc in spec_docs:
        label = f"[Spec chunk {doc.metadata.get('chunk_index', '?')}]"
        text_parts.append(f"{label}\n{doc.page_content}")
    text_context = "\n\n---\n\n".join(text_parts)

    # If the query targets a specific page (pinned), send ONLY that page's 9 tiles.
    # Otherwise send the top-2 retrieved pages (2 × 9 = 18 images).
    is_page_specific = any(d.metadata.get("pinned") for d in blueprint_docs)
    pages_to_render = blueprint_docs[:1] if is_page_specific else blueprint_docs[:2]

    image_parts = []
    for doc in pages_to_render:
        page_num = doc.metadata.get("page_number", "?")
        image_path = doc.metadata.get("image_path", "")
        if not image_path:
            text_parts.append(f"[Blueprint page {page_num}]\n[Image not available]")
            continue
        image_parts.extend(_page_quadrants(image_path, page_num))

    # Extra instruction injected only for page-specific queries
    page_specific_rule = ""
    if is_page_specific:
        pinned_page = next(
            (d.metadata.get("page_number", "?") for d in blueprint_docs if d.metadata.get("pinned")),
            "?"
        )
        page_specific_rule = (
            f"\n\nPAGE-SPECIFIC QUERY — the user is asking about page {pinned_page}. "
            "You MUST go through ALL 9 tiles in order: top-left → top-center → top-right → "
            "middle-left → middle-center → middle-right → bottom-left → bottom-center → bottom-right. "
            "Each tile gets its own section. Never combine or skip tiles. "
            "Even if a tile looks sparse, report every element visible in it — "
            "every number, letter, line, hatch, symbol, and annotation."
        )

    system_prompt = (
        "You are a construction document assistant specializing in exhaustive blueprint analysis. "
        "Each blueprint page is sent as nine high-resolution tiles in a 3×3 grid: "
        "'top-left', 'top-center', 'top-right', 'middle-left', 'middle-center', 'middle-right', "
        "'bottom-left', 'bottom-center', 'bottom-right'. "
        "\n\n"
        "REGARDLESS of how short or simple the user's question is, you MUST always:\n"
        "(1) Process every tile — never skip a tile even if it looks empty.\n"
        "(2) Transcribe every word of text verbatim — every label, note, callout, title, and annotation.\n"
        "(3) Read every number, dimension, measurement, fraction, scale, and unit exactly as written "
        "— pay special attention to dimension strings like 3'-6\", 1/8\"=1'-0\", and fractions.\n"
        "(4) Describe every diagram, symbol, line type, hatch pattern, arrow, and leader line "
        "and exactly what each leader line points to.\n"
        "(5) Read every row and column of any table or schedule completely.\n"
        "(6) Read any title block, drawing number, revision cloud, sheet name, scale bar, "
        "north arrow, stamp, license number, or project address.\n"
        "\n"
        "Never summarize or skip details because the question was short. "
        "A question like 'explain page 4' or 'what is on page 4' demands the same exhaustive "
        "tile-by-tile response as an explicit detailed request."
        + page_specific_rule +
        "\n\n"
        "FORMAT YOUR ENTIRE ANSWER AS STRUCTURED MARKDOWN using this pattern for each diagram:\n"
        "\n"
        "## [Diagram Name / Drawing Title] — [Tile Location]\n"
        "### Geometric & Spatial Data\n"
        "| Element | Value | Unit |\n"
        "|---------|-------|------|\n"
        "| dimension name | exact number | ft / in / deg |\n"
        "### Components & Annotations\n"
        "- **[label]**: what it is, what the leader line points to\n"
        "### Notes & Code References\n"
        "- verbatim note text\n"
        "### Schedule / Table (if present)\n"
        "| Col A | Col B | Col C |\n"
        "|-------|-------|-------|\n"
        "| val   | val   | val   |\n"
        "\n"
        "End every response with a '## Title Block' section listing: drawing number, sheet title, "
        "project name, address, issue date, drawn by, checked by, scale, and firm name.\n"
        "Cite every finding with its tile, e.g. [Blueprint page 4 — middle-center tile].\n"
        'Respond with a JSON object with exactly these keys: '
        '"answer" (string — the full markdown text), '
        '"cited_sources" (list of {label, document_id, page_or_chunk}), '
        '"confidence_score" (float 0-1).'
    )

    if image_parts:
        user_content = []
        if text_context:
            user_content.append({"type": "text", "text": f"Text context:\n{text_context}\n\n"})
        user_content.extend(image_parts)
        user_content.append({
            "type": "text",
            "text": f"\nQuestion: {state['query']}\n\nJSON response:",
        })
        response = get_llm().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
    else:
        prompt = (
            f"{system_prompt}\n\nQuestion: {state['query']}\n\n"
            f"Context:\n{text_context}\n\nJSON response:"
        )
        response = get_llm().invoke(prompt)

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
        answer = parsed.get("answer", "No answer generated.")
        cited_sources = parsed.get("cited_sources", [])
        confidence_score = float(parsed.get("confidence_score", state["confidence"]))
    except Exception:
        answer = content
        cited_sources = []
        confidence_score = state["confidence"]

    return {
        "answer": answer,
        "cited_sources": cited_sources,
        "confidence": confidence_score,
    }


def route_retriever(state: RAGState) -> list[str]:
    qt = state["query_type"]
    if qt == "blueprint":
        return ["blueprint_retrieve"]
    if qt == "spec":
        return ["text_retrieve"]
    return ["blueprint_retrieve", "text_retrieve"]


# ── graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(RAGState)
builder.add_node("router", router_node)
builder.add_node("blueprint_retrieve", blueprint_retrieve_node)
builder.add_node("text_retrieve", text_retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("rewrite", rewrite_query_node)
builder.add_node("widen", widen_query_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router", route_retriever, ["blueprint_retrieve", "text_retrieve"]
)
builder.add_edge("blueprint_retrieve", "rerank")
builder.add_edge("text_retrieve", "rerank")
builder.add_conditional_edges(
    "rerank",
    confidence_check_node,
    {"generate": "generate", "rewrite": "rewrite", "widen": "widen"},
)
builder.add_conditional_edges(
    "rewrite",
    lambda s: "blueprint_retrieve"
    if s["query_type"] == "blueprint"
    else ("text_retrieve" if s["query_type"] == "spec" else "blueprint_retrieve"),
    ["blueprint_retrieve", "text_retrieve"],
)
builder.add_conditional_edges(
    "widen",
    route_retriever,
    ["blueprint_retrieve", "text_retrieve"],
)
builder.add_edge("generate", END)

graph = builder.compile()


def run_query(query: str, document_ids: list[str]) -> dict:
    initial_state: RAGState = {
        "query": query,
        "document_ids": document_ids,
        "query_type": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "confidence": 0.0,
        "answer": "",
        "cited_sources": [],
        "retry_count": 0,
        "rewritten_query": None,
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "cited_sources": result["cited_sources"],
        "confidence_score": result["confidence"],
    }
