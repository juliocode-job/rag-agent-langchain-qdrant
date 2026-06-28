# RAG Agent — LangChain + Qdrant + FastAPI Streaming

Production-ready RAG (Retrieval-Augmented Generation) agent using **LangChain**, **Qdrant Cloud**, **FastAPI** with **streaming** support.

## Stack

| Layer | Tool |
|---|---|
| Vector DB | Qdrant Cloud (Free Tier) |
| Orchestration | LangChain |
| API | FastAPI |
| Streaming | Server-Sent Events (SSE) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Language | Python 3.11+ |

## Features

- `POST /ingest` — upload and chunk documents into Qdrant
- `POST /chat` — RAG chat with source documents returned
- `POST /chat/stream` — streaming response via SSE
- `GET /health` — health check
- Configurable chunking strategy
- Async FastAPI + LangChain LCEL pipeline
- `.env` based configuration

## Quickstart

```bash
# 1. Clone
git clone https://github.com/juliocode-job/rag-agent-langchain-qdrant.git
cd rag-agent-langchain-qdrant

# 2. Create venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your Qdrant Cloud URL, API key, and OpenAI key

# 5. Run
uvicorn app.main:app --reload
```

Docs available at `http://localhost:8000/docs`

## Project Structure

```
rag-agent-langchain-qdrant/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings via pydantic-settings
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── services/
│   │   ├── ingest.py        # Document ingestion pipeline
│   │   ├── retriever.py     # Qdrant retriever setup
│   │   └── chain.py         # LangChain RAG chain (LCEL)
│   └── schemas/
│       └── models.py        # Pydantic request/response models
├── tests/
│   └── test_routes.py
├── .env.example
├── requirements.txt
└── README.md
```

## Environment Variables

See `.env.example` for all required variables.

```env
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
COLLECTION_NAME=rag_documents
```
