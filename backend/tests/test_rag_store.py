from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.app.rag.ingestion import documents_from_entries
from backend.app.rag.store import NeonKnowledgeStore, _format_vector


class FakeEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        document_vectors: list[list[float]] | None = None,
        query_vector: list[float] | None = None,
    ) -> None:
        self.document_vectors = document_vectors or []
        self.query_vector = query_vector or []
        self.document_inputs: list[list[str]] = []
        self.query_inputs: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs.append(texts)
        return self.document_vectors

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return self.query_vector


class FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class FakeCursor:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple[str, Sequence[Sequence[Any]]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def executemany(
        self,
        query: str,
        rows: Sequence[Sequence[Any]],
    ) -> None:
        self.executemany_calls.append((query, rows))


class FakeConnection:
    def __init__(self, search_rows: list[tuple[Any, ...]]) -> None:
        self.search_rows = search_rows
        self.execute_calls: list[tuple[str, object | None]] = []
        self.fake_cursor = FakeCursor()

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(
        self,
        query: str,
        params: object | None = None,
    ) -> FakeResult:
        self.execute_calls.append((query, params))
        return FakeResult(self.search_rows if "SELECT" in query else [])

    def cursor(self) -> FakeCursor:
        return self.fake_cursor


class FakeConnector:
    def __init__(self, search_rows: list[tuple[Any, ...]] | None = None) -> None:
        self.search_rows = search_rows or []
        self.connections: list[FakeConnection] = []

    def __call__(self, database_url: str) -> FakeConnection:
        assert database_url == "postgresql://test"
        connection = FakeConnection(self.search_rows)
        self.connections.append(connection)
        return connection


def test_documents_from_entries_builds_langchain_documents() -> None:
    documents = documents_from_entries(
        [
            {
                "content": "  Haircuts start at $35. ",
                "source": " service-catalog ",
                "category": " pricing ",
            },
            {
                "content": "Appointments require 24 hours notice.",
                "source": "policy-manual",
            },
        ]
    )

    assert documents == [
        Document(
            page_content="Haircuts start at $35.",
            metadata={"source": "service-catalog", "category": "pricing"},
        ),
        Document(
            page_content="Appointments require 24 hours notice.",
            metadata={"source": "policy-manual", "category": "general"},
        ),
    ]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"content": "not a list"}, "JSON array"),
        (["not an object"], "entry 0 must be an object"),
        ([{"content": "", "source": "catalog"}], "content must be text"),
        ([{"content": "Policy", "source": ""}], "source must be text"),
        (
            [{"content": "Policy", "source": "manual", "category": 7}],
            "category must be text",
        ),
    ],
)
def test_documents_from_entries_rejects_invalid_records(
    entries: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        documents_from_entries(entries)


def test_store_indexes_documents_with_parameterized_upsert(monkeypatch: Any) -> None:
    embeddings = FakeEmbeddings(document_vectors=[[0.1, 0.2, 0.3]])
    connector = FakeConnector()
    monkeypatch.setattr("backend.app.rag.store._connect", connector)
    store = NeonKnowledgeStore(
        database_url="postgresql://test",
        embeddings=embeddings,
        embedding_dimensions=3,
    )

    indexed_count = store.index_documents(
        [
            Document(
                page_content="  Haircuts start at $35.  ",
                metadata={"source": "catalog", "category": "pricing"},
            ),
            Document(page_content="   "),
        ]
    )

    assert indexed_count == 1
    assert embeddings.document_inputs == [["Haircuts start at $35."]]
    assert len(connector.connections) == 2
    schema_queries = [query for query, _ in connector.connections[0].execute_calls]
    assert any(
        "CREATE EXTENSION IF NOT EXISTS vector" in query
        for query in schema_queries
    )
    assert any("USING hnsw" in query for query in schema_queries)

    upsert_query, rows = connector.connections[1].fake_cursor.executemany_calls[0]
    assert "ON CONFLICT (content_hash) DO UPDATE" in upsert_query
    assert "%s::vector" in upsert_query
    assert len(rows) == 1
    row = rows[0]
    assert row[2:] == (
        "catalog",
        "pricing",
        "Haircuts start at $35.",
        "[0.1,0.2,0.3]",
    )
    assert len(row[1]) == 64


def test_store_searches_by_cosine_distance(monkeypatch: Any) -> None:
    embeddings = FakeEmbeddings(query_vector=[0.3, 0.2, 0.1])
    connector = FakeConnector(
        search_rows=[("Policy text", "policy-manual", "policy", 0.87)]
    )
    monkeypatch.setattr("backend.app.rag.store._connect", connector)
    store = NeonKnowledgeStore(
        database_url="postgresql://test",
        embeddings=embeddings,
        embedding_dimensions=3,
        top_k=2,
    )

    documents = store.search("  What is the refund policy?  ")

    assert embeddings.query_inputs == ["What is the refund policy?"]
    assert documents == [
        Document(
            page_content="Policy text",
            metadata={
                "source": "policy-manual",
                "category": "policy",
                "similarity": 0.87,
            },
        )
    ]
    search_query, params = connector.connections[1].execute_calls[0]
    assert "embedding <=> %s::vector" in search_query
    assert params == ("[0.3,0.2,0.1]", "[0.3,0.2,0.1]", 2)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.1, 0.2], "dimension mismatch"),
        ([0.1, float("inf"), 0.3], "must be finite"),
    ],
)
def test_vector_format_rejects_invalid_embeddings(
    vector: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _format_vector(vector, expected_dimensions=3)
