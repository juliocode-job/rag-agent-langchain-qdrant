from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.services.retriever import get_vector_store
from app.config import settings


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions based on the provided context.
Always base your answers on the context below. If you don't know the answer from the context, say so clearly.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """Returns a LCEL RAG chain that also exposes source_documents."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openai_api_key,
    )

    # Return answer + source documents
    rag_chain_with_sources = RunnableParallel(
        {"source_documents": retriever, "question": RunnablePassthrough()}
    ).assign(
        context=lambda x: _format_docs(x["source_documents"]),
    ).assign(
        answer=RAG_PROMPT | llm | StrOutputParser()
    )

    return rag_chain_with_sources


def get_rag_chain_stream():
    """Returns a streaming LCEL chain (tokens only, no source docs)."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openai_api_key,
        streaming=True,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
