from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class _PreparedDocument:
    content: str
    business_type: str
    source: str
    category: str
    content_hash: str


@dataclass
class NeonKnowledgeStore:
    database_url: str
    embeddings: Embeddings
    embedding_dimensions: int = 1536
    top_k: int = 4
    _schema_ready: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.database_url.strip():
            raise ValueError("database_url must not be empty")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return

        with _connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id UUID PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    business_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.embedding_dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE knowledge_chunks
                ADD COLUMN IF NOT EXISTS business_type TEXT NOT NULL DEFAULT 'general'
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS knowledge_chunks_business_type_idx
                ON knowledge_chunks (business_type)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_hnsw_idx
                ON knowledge_chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )

        self._schema_ready = True

    def index_documents(self, documents: list[Document]) -> int:
        prepared_documents = [
            prepared
            for document in documents
            if (prepared := _prepare_document(document)) is not None
        ]
        if not prepared_documents:
            return 0

        vectors = self.embeddings.embed_documents(
            [document.content for document in prepared_documents]
        )
        if len(vectors) != len(prepared_documents):
            raise RuntimeError("Document and embedding counts do not match")

        rows = [
            (
                uuid4(),
                document.content_hash,
                document.business_type,
                document.source,
                document.category,
                document.content,
                _format_vector(vector, self.embedding_dimensions),
            )
            for document, vector in zip(prepared_documents, vectors, strict=True)
        ]

        self.ensure_schema()
        with _connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO knowledge_chunks
                        (
                            id,
                            content_hash,
                            business_type,
                            source,
                            category,
                            content,
                            embedding
                        )
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        business_type = EXCLUDED.business_type,
                        source = EXCLUDED.source,
                        category = EXCLUDED.category,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    rows,
                )

        return len(rows)

    def search(self, query: str, business_type: str | None = None) -> list[Document]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Retrieval query must not be empty")

        query_vector = _format_vector(
            self.embeddings.embed_query(cleaned_query),
            self.embedding_dimensions,
        )
        self.ensure_schema()

        cleaned_business_type = business_type.strip() if business_type else None
        with _connect(self.database_url) as connection:
            if cleaned_business_type:
                rows = connection.execute(
                    """
                    SELECT
                        content,
                        business_type,
                        source,
                        category,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM knowledge_chunks
                    WHERE business_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        cleaned_business_type,
                        query_vector,
                        self.top_k,
                    ),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        content,
                        business_type,
                        source,
                        category,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM knowledge_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, query_vector, self.top_k),
                ).fetchall()

        return [
            Document(
                page_content=content,
                metadata={
                    "business_type": business_type,
                    "source": source,
                    "category": category,
                    "similarity": float(similarity),
                },
            )
            for content, business_type, source, category, similarity in rows
        ]


def _prepare_document(document: Document) -> _PreparedDocument | None:
    content = document.page_content.strip()
    if not content:
        return None

    source = str(document.metadata.get("source") or "business_knowledge").strip()
    category = str(document.metadata.get("category") or "general").strip()
    business_type = str(document.metadata.get("business_type") or "general").strip()
    fingerprint = f"{business_type}\0{source}\0{category}\0{content}".encode("utf-8")

    return _PreparedDocument(
        content=content,
        business_type=business_type or "general",
        source=source or "business_knowledge",
        category=category or "general",
        content_hash=sha256(fingerprint).hexdigest(),
    )


def _connect(database_url: str) -> Any:
    import psycopg

    return psycopg.connect(database_url)


def _format_vector(values: list[float], expected_dimensions: int) -> str:
    if len(values) != expected_dimensions:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected {expected_dimensions}, received {len(values)}"
        )

    normalized_values = [float(value) for value in values]
    if not all(isfinite(value) for value in normalized_values):
        raise ValueError("Embedding values must be finite")

    return "[" + ",".join(str(value) for value in normalized_values) + "]"
