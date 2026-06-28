# RAG Agent — LangChain + Qdrant + FastAPI Streaming

Production-ready RAG (Retrieval-Augmented Generation) agent using **LangChain**, **Qdrant Cloud**, **FastAPI** with **streaming** support.

## Stack

| Layer | Tool |
|---|---|
| Vector DB | Qdrant Cloud (Free Tier) |
| Orchestration | LangChain LCEL |
| API | FastAPI |
| Streaming | Server-Sent Events (SSE) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | Anthropic Claude (`claude-3-5-haiku-20241022`) |
| Language | Python 3.11+ |

## Features

- `POST /api/v1/ingest` — upload e chunking de documentos (PDF, TXT, MD) para o Qdrant
- `POST /api/v1/chat` — RAG com resposta + source documents
- `POST /api/v1/chat/stream` — streaming de tokens via SSE
- `GET /health` — health check
- Configuração via `.env` (pydantic-settings)
- Pipeline assíncrono FastAPI + LangChain LCEL
- Criação automática da collection Qdrant no startup

## Quickstart

```bash
# 1. Clone
git clone https://github.com/juliocode-job/rag-agent-langchain-qdrant.git
cd rag-agent-langchain-qdrant

# 2. Crie o venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instale
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Preencha QDRANT_URL, QDRANT_API_KEY, ANTHROPIC_API_KEY e OPENAI_API_KEY

# 5. Rode
uvicorn app.main:app --reload
```

Swagger em `http://localhost:8000/docs`

## Variáveis de Ambiente

```env
# Qdrant Cloud
QDRANT_URL=https://<cluster-id>.cloud.qdrant.io:6333
QDRANT_API_KEY=sua_qdrant_api_key

# Anthropic (LLM)
ANTHROPIC_API_KEY=sua_anthropic_api_key

# OpenAI (embeddings)
OPENAI_API_KEY=sua_openai_api_key

# Modelo Anthropic
LLM_MODEL=claude-3-5-haiku-20241022
```

> Para usar Claude Sonnet, altere `LLM_MODEL=claude-sonnet-4-5` no `.env`.

## Project Structure

```
rag-agent-langchain-qdrant/
├── app/
│   ├── main.py              # FastAPI entry point + lifespan
│   ├── config.py            # Settings via pydantic-settings
│   ├── api/
│   │   └── routes.py        # Endpoints (ingest, chat, chat/stream)
│   ├── services/
│   │   ├── ingest.py        # Document ingestion pipeline
│   │   ├── retriever.py     # Qdrant vector store + embeddings
│   │   └── chain.py         # LangChain LCEL RAG chain (Anthropic)
│   └── schemas/
│       └── models.py        # Pydantic request/response models
├── tests/
│   └── test_routes.py
├── .env.example
├── requirements.txt
└── README.md
```

## Modelos Anthropic Disponíveis

| Modelo | Uso recomendado |
|---|---|
| `claude-3-5-haiku-20241022` | Rápido, barato — padrão para RAG |
| `claude-sonnet-4-5` | Qualidade superior para respostas complexas |
| `claude-opus-4-5` | Máxima capacidade de raciocínio |
