from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.models import ChatRequest, ChatResponse, IngestResponse
from app.services.ingest import ingest_documents
from app.services.chain import get_rag_chain, get_rag_chain_stream

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest(
    files: list[UploadFile] = File(...),
):
    """Ingest documents (PDF, TXT) into the Qdrant vector store."""
    try:
        total_chunks = await ingest_documents(files)
        return IngestResponse(
            message=f"Successfully ingested {len(files)} file(s).",
            chunks_created=total_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Single-turn RAG chat — returns answer and source documents."""
    try:
        chain = get_rag_chain()
        result = await chain.ainvoke({"question": request.question})
        return ChatResponse(
            answer=result["answer"],
            sources=[doc.metadata for doc in result.get("source_documents", [])],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """Streaming RAG chat via Server-Sent Events (SSE)."""

    async def event_generator():
        try:
            chain = get_rag_chain_stream()
            async for chunk in chain.astream({"question": request.question}):
                if isinstance(chunk, str):
                    yield f"data: {chunk}\n\n"
                elif hasattr(chunk, "content"):
                    yield f"data: {chunk.content}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
