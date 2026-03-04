import json
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

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


def router_node(state: RAGState) -> dict:
    prompt = (
        f"Classify this query into one of three categories:\n"
        f"- 'blueprint': questions about diagrams, floor plans, layouts, visual elements\n"
        f"- 'spec': questions about text specifications, codes, requirements\n"
        f"- 'cross_document': questions that span both visual and text sources\n\n"
        f"Query: {state['query']}\n\n"
        f"Reply with exactly one word: blueprint, spec, or cross_document"
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
    return {"retrieved_docs": docs}  # reducer appends automatically


def text_retrieve_node(state: RAGState) -> dict:
    q = state.get("rewritten_query") or state["query"]
    retriever = TextRetriever(document_ids=state["document_ids"])
    docs = retriever.invoke(q)
    return {"retrieved_docs": docs}  # reducer appends automatically


def rerank_node(state: RAGState) -> dict:
    q = state.get("rewritten_query") or state["query"]
    reranked = rerank_documents(q, state["retrieved_docs"])
    confidence = compute_confidence(reranked)
    return {"reranked_docs": reranked, "confidence": confidence}


def confidence_check_node(state: RAGState) -> str:
    if state["confidence"] >= 0.4 or state["retry_count"] >= 2:
        return "generate"
    return "rewrite"


def rewrite_query_node(state: RAGState) -> dict:
    prompt = (
        f"The following query returned low-confidence results. "
        f"Rewrite it to be more specific and detailed for a construction document search.\n\n"
        f"Original: {state['query']}\n\n"
        f"Return only the rewritten query, nothing else."
    )
    response = get_llm().invoke(prompt)
    return {
        "rewritten_query": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: RAGState) -> dict:
    context_parts = []
    for doc in state["reranked_docs"]:
        meta = doc.metadata
        source_label = (
            f"[Blueprint page {meta.get('page_number', '?')}]"
            if meta.get("source") == "blueprint"
            else f"[Spec chunk {meta.get('chunk_index', '?')}]"
        )
        context_parts.append(f"{source_label}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a construction document assistant. Answer the question using only the provided context.
Cite sources by referring to the labels in square brackets (e.g. [Blueprint page 3] or [Spec chunk 5]).

Question: {state['query']}

Context:
{context}

Respond with a JSON object with exactly these keys:
- "answer": your detailed answer as a string
- "cited_sources": a list of objects, each with keys "label", "document_id", "page_or_chunk"
- "confidence_score": a float between 0 and 1

JSON response:"""

    response = get_llm().invoke(prompt)
    content = response.content.strip()

    # Strip markdown code fences if present
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


# Build graph
builder = StateGraph(RAGState)
builder.add_node("router", router_node)
builder.add_node("blueprint_retrieve", blueprint_retrieve_node)
builder.add_node("text_retrieve", text_retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("rewrite", rewrite_query_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_retriever, ["blueprint_retrieve", "text_retrieve"])
builder.add_edge("blueprint_retrieve", "rerank")
builder.add_edge("text_retrieve", "rerank")
builder.add_conditional_edges(
    "rerank",
    confidence_check_node,
    {"generate": "generate", "rewrite": "rewrite"},
)
builder.add_conditional_edges(
    "rewrite",
    lambda s: "blueprint_retrieve" if s["query_type"] == "blueprint" else (
        "text_retrieve" if s["query_type"] == "spec" else "blueprint_retrieve"
    ),
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
