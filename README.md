# PlanSmartAI MVP

Upload construction PDFs — blueprints or spec documents — and ask questions in plain English. The system extracts text and visual embeddings, retrieves the most relevant pages or chunks, reranks them, and returns a cited answer with a confidence score.

## Stack

- **FastAPI** — REST backend (upload, query, status)
- **Chainlit** — chat UI
- **Qdrant** — vector store (ColQwen2 multi-vector for blueprints, BGE-M3 hybrid for specs)
- **PostgreSQL** — document and chunk metadata
- **LangGraph** — RAG pipeline with router, retriever, reranker, and generator nodes
- **ColQwen2** — visual multi-vector embeddings for blueprint pages
- **BGE-M3 + BGE-Reranker-v2-m3** — dense/sparse hybrid embeddings + reranking for spec text
- **NuMarkdown-8B-Thinking** (HuggingFace) — extracts markdown from blueprint page images
- **OpenRouter** — LLM gateway (Claude 3.5 Sonnet by default)

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd plansmartai-Numarkdown

# 2. Create .env from example and fill in your keys
cp .env.example .env
# Edit .env: add OPENROUTER_API_KEY and HF_API_KEY

# 3. Start Qdrant + PostgreSQL
docker compose up -d

# 4. Create a Python 3.11 virtualenv and install deps
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Start the API
uvicorn main:app --reload --port 8000

# 6. Start the chat UI (new terminal, venv activated)
chainlit run chainlit_app.py --port 8001
```

Open http://localhost:8001, upload a PDF, and start asking questions.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload a PDF (form: `file`, `file_type`) |
| GET | `/documents` | List all documents |
| GET | `/documents/{id}/status` | Poll processing status |
| POST | `/query` | Ask a question (`query`, `document_ids`) |

## Known limitations

- NuMarkdown-8B-Thinking on HuggingFace Inference API can be slow (cold starts). The client retries 503s automatically.
- ColQwen2 stores the first patch vector per page for Qdrant search; full late-interaction scoring is applied at rerank time.
- No authentication on API endpoints.
- BGE-M3 model download (~2 GB) happens on first run.

## Coming next

- Multi-document cross-reference answers
- RAGAS-based evaluation harness
- Authentication and per-user document isolation
- Self-hosted NuMarkdown inference container
