import os

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import OpenAIEmbeddings

from backend.app.rag.store import NeonKnowledgeStore


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_TOP_K = 4

RAGRetriever = Runnable[str, list[Document]]


def create_neon_knowledge_store(
    *,
    database_url: str | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    top_k: int | None = None,
) -> NeonKnowledgeStore:
    selected_database_url = database_url or os.getenv("DATABASE_URL", "")
    if not selected_database_url.strip():
        raise RuntimeError("DATABASE_URL is not configured")

    selected_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not selected_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is not configured")

    selected_dimensions = (
        _positive_int_from_env(
            "OPENAI_EMBEDDING_DIMENSIONS",
            DEFAULT_EMBEDDING_DIMENSIONS,
        )
        if embedding_dimensions is None
        else _require_positive("embedding_dimensions", embedding_dimensions)
    )
    selected_top_k = (
        _positive_int_from_env("RAG_TOP_K", DEFAULT_TOP_K)
        if top_k is None
        else _require_positive("top_k", top_k)
    )
    selected_model = embedding_model or os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    )
    embeddings = OpenAIEmbeddings(
        model=selected_model,
        dimensions=selected_dimensions,
        api_key=selected_api_key,
        max_retries=2,
    )

    return NeonKnowledgeStore(
        database_url=selected_database_url,
        embeddings=embeddings,
        embedding_dimensions=selected_dimensions,
        top_k=selected_top_k,
    )


def create_neon_retriever(
    store: NeonKnowledgeStore | None = None,
) -> RAGRetriever:
    knowledge_store = store or create_neon_knowledge_store()
    return RunnableLambda(knowledge_store.search)


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc

    return _require_positive(name, value)


def _require_positive(name: str, value: int) -> int:
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")

    return value
