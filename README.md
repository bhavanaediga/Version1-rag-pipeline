# PlanSmartAI — Construction Document RAG Pipeline

A multimodal Retrieval-Augmented Generation (RAG) system purpose-built for construction documents. Upload engineering blueprints or specification PDFs, ask questions in plain English, and get precise cited answers with confidence scores — including exact dimension callouts, table values, annotations, and diagram descriptions extracted at high resolution.

---

## How It Works

The system runs two separate ingestion pipelines depending on document type, indexes embeddings into a vector database, and routes every query through a LangGraph agentic graph before generating a structured markdown answer via Claude 3.5 Sonnet (vision).

**Blueprint pages** are converted to JPEG images, embedded with ColQwen2 (a visual multi-vector model), and stored in Qdrant with MaxSim comparator. At query time each retrieved page is split into a **3×3 grid of 9 tiles**, each upscaled to 2048px wide, and sent to Claude 3.5 Sonnet as 18 high-resolution images per LLM call — enabling it to read small dimension strings, fractions, table values, and annotations that a full-page image would miss.

**Spec documents** are parsed into markdown by Docling, chunked, and dual-embedded with BGE-M3 (dense 1024-dim + sparse BM25-style lexical). Qdrant's RRF fusion merges both at query time for hybrid semantic + keyword retrieval.

---

## Architecture

```
User (Chainlit UI :8001)
        │ HTTP
        ▼
FastAPI Backend (:8000)
        │
        ├── /upload  → Background ingestion task
        │       ├── Blueprint: pdf2image → ColQwen2 → Qdrant (blueprint_visual)
        │       └── Spec:      Docling   → BGE-M3   → Qdrant (text_hybrid)
        │
        └── /query   → LangGraph StateGraph
                ├── router_node        — classify: blueprint | spec | cross_document
                ├── blueprint_retrieve — ColQwen2 MaxSim OR direct page lookup
                ├── text_retrieve      — BGE-M3 RRF hybrid (dense + sparse)
                ├── rerank_node        — BGE-Reranker-v2-M3 cross-encoder
                ├── confidence_check   — route to generate / rewrite / widen
                ├── rewrite_node       — Claude rewrites low-confidence queries
                ├── widen_node         — fallback to cross_document on zero results
                └── generate_node      — 3×3 tile vision → Claude 3.5 Sonnet

Databases (Docker):
  PostgreSQL :5432  — document metadata, blueprint_pages, text_chunks
  Qdrant     :6333  — blueprint_visual (ColQwen2 multi-vector), text_hybrid (BGE-M3 hybrid)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| REST API | FastAPI + Uvicorn |
| Chat UI | Chainlit 2.x |
| Agent Orchestration | LangGraph |
| LLM Abstraction | LangChain + langchain-openai |
| LLM (Vision + Generation) | Claude 3.5 Sonnet via OpenRouter |
| Visual Embedding | ColQwen2 (`vidore/colqwen2-v1.0`) via colpali-engine |
| Text Embedding | BGE-M3 (`BAAI/bge-m3`) via FlagEmbedding |
| Reranker | BGE-Reranker-v2-M3 (`BAAI/bge-reranker-v2-m3`) via FlagEmbedding |
| Vector Database | Qdrant 1.17 (Docker) |
| Relational Database | PostgreSQL 15 (Docker) |
| ORM | SQLAlchemy 2.0 + psycopg2 |
| PDF → Image | pdf2image (poppler/pdftocairo) + pypdfium2 |
| PDF → Markdown | Docling 2.5 |
| Image Processing | Pillow (crop, resize, LANCZOS, base64 encode) |
| Deep Learning Runtime | PyTorch 2.6.0 |
| Async HTTP | httpx |
| Containerization | Docker + Docker Compose |

---

## Qdrant Collections

**`blueprint_visual`**
- Vector: `colqwen2` — 128-dim, Cosine, MultiVector with MAX_SIM comparator
- One point per blueprint page
- Payload: `document_id`, `page_number`, `image_path`

**`text_hybrid`**
- Dense vector: `dense` — 1024-dim, Cosine
- Sparse vector: `sparse` — BM25-style lexical weights
- Fusion: Reciprocal Rank Fusion (RRF) at query time

---

## PostgreSQL Schema

**`documents`** — one row per uploaded PDF (`id`, `file_name`, `file_type`, `processing_status`, `progress_detail`, `uploaded_at`)

**`blueprint_pages`** — one row per page (`document_id`, `page_number`, `image_path`, `extracted_text`, `qdrant_point_id`)

**`text_chunks`** — one row per text chunk (`document_id`, `chunk_index`, `content`, `section_header`, `qdrant_point_id`)

---

## 3×3 Vision Tiling

Each retrieved blueprint page is divided into a 3×3 grid of 9 tiles. Each tile covers 1/3 of the page width and height. Every tile is upscaled to 2048px wide using LANCZOS resampling and JPEG-encoded at quality 92. Tiles are labeled: `top-left`, `top-center`, `top-right`, `middle-left`, `middle-center`, `middle-right`, `bottom-left`, `bottom-center`, `bottom-right`. The top 2 retrieved pages × 9 tiles = 18 images are sent per LLM call as interleaved text labels and base64 image_url blocks.

This resolves small annotation text, dimension strings (e.g. `3'-6"`), table values, schedule rows, and leader line targets that are unreadable at full-page resolution.

---

## Reranking Logic

Retrieved documents are sorted into three tiers before being passed to the generator:

1. **Pinned** — if the query contains an explicit page number, that page is fetched directly from PostgreSQL (bypasses Qdrant entirely) and assigned a fixed score of 10.0, placed first.
2. **Text-rich** — documents with more than 30 characters of extracted text are scored by BGE-Reranker-v2-M3 as query-passage cross-encoder pairs.
3. **Visual-only** — documents with sparse extracted text fall back to their ColQwen2 MaxSim score, remapped to the [-10, 10] scale.

Confidence is the average of the top-3 reranker scores normalised to [0, 1]. Queries below 0.4 confidence trigger an automatic rewrite via Claude; zero results trigger a widening to cross-document search.

---

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/bhavanaediga/Version1-rag-pipeline.git
cd Version1-rag-pipeline
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Start databases

```bash
docker compose up -d
# Starts Qdrant on :6333 and PostgreSQL on :5432
```

### 3. Install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> First run downloads BGE-M3 (~2 GB) and ColQwen2 (~5 GB) from HuggingFace on first use.

### 4. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

### 5. Start the chat UI (new terminal)

```bash
source venv/bin/activate
chainlit run chainlit_app.py --port 8001
```

Open [http://localhost:8001](http://localhost:8001), upload a construction PDF, and ask questions.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload PDF (`file`, `file_type`: blueprint or spec) |
| `GET` | `/documents` | List all documents with processing status |
| `GET` | `/documents/{id}/status` | Poll ingestion progress |
| `POST` | `/query` | Run RAG query (`query`, `document_ids[]`) |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "give me details of page 24", "document_ids": ["your-doc-uuid"]}'
```

Response includes `answer` (structured markdown), `cited_sources` (with tile labels like `Blueprint page 24 — middle-center tile`), and `confidence_score`.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
OPENROUTER_API_KEY   # OpenRouter key — used to call Claude 3.5 Sonnet
HF_API_KEY           # HuggingFace key — used for model downloads
POSTGRES_URL         # defaults to postgresql://plansmartai:plansmartai@localhost:5432/plansmartai
QDRANT_URL           # defaults to http://localhost:6333
```

---

## Known Constraints

- ColQwen2 and BGE-M3 run on CPU (MPS disabled — bfloat16 matmul shape incompatibility on Apple Silicon). Expect 25–60s per query on CPU.
- BGE-M3 is ~2 GB and ColQwen2 is ~5 GB — downloaded once on first use.
- No authentication on API endpoints — intended for local/internal use.
- `torch` and `torchvision` must be installed at matching versions (`2.6.0` / `0.21.0`).
