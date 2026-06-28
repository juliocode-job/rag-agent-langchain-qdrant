from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )


async def ensure_collection() -> None:
    """Create Qdrant collection if it does not exist yet."""
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if settings.collection_name not in names:
        await client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Collection '{settings.collection_name}' created.")
    else:
        print(f"[Qdrant] Collection '{settings.collection_name}' already exists.")
    await client.close()
