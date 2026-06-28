from langchain_anthropic import ChatAnthropic
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


def _get_llm(streaming: bool = False) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
        streaming=streaming,
    )


def get_rag_chain():
    """LCEL RAG chain — returns answer + source_documents."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    rag_chain_with_sources = (
        RunnableParallel(
            {"source_documents": retriever, "question": RunnablePassthrough()}
        )
        .assign(context=lambda x: _format_docs(x["source_documents"]))
        .assign(answer=RAG_PROMPT | _get_llm() | StrOutputParser())
    )

    return rag_chain_with_sources


def get_rag_chain_stream():
    """Streaming LCEL chain — yields tokens via SSE."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | _get_llm(streaming=True)
        | StrOutputParser()
    )

    return chain
