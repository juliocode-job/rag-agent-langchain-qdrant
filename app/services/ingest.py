import tempfile
import os
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.services.retriever import get_vector_store, get_embeddings
from app.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


async def ingest_documents(files: list[UploadFile]) -> int:
    """Load, chunk and upsert documents into Qdrant."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    all_chunks = []

    for file in files:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()
            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = file.filename

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        finally:
            os.unlink(tmp_path)

    if all_chunks:
        vector_store = get_vector_store()
        vector_store.add_documents(all_chunks)

    return len(all_chunks)
