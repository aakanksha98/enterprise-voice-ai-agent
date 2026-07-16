import json
from pathlib import Path

from langchain_core.documents import Document

from backend.app.rag.retriever import create_neon_knowledge_store
from backend.app.rag.store import NeonKnowledgeStore


def load_knowledge_documents(path: Path) -> list[Document]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return documents_from_entries(entries)


def documents_from_entries(entries: object) -> list[Document]:
    if not isinstance(entries, list):
        raise ValueError("Knowledge file must contain a JSON array")

    documents: list[Document] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Knowledge entry {index} must be an object")

        content = _required_string(entry, "content", index)
        source = _required_string(entry, "source", index)
        category_value = entry.get("category", "general")
        if not isinstance(category_value, str) or not category_value.strip():
            raise ValueError(f"Knowledge entry {index} category must be text")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "category": category_value.strip(),
                },
            )
        )

    return documents


def index_knowledge_file(
    path: Path,
    store: NeonKnowledgeStore | None = None,
) -> int:
    knowledge_store = store or create_neon_knowledge_store()
    return knowledge_store.index_documents(load_knowledge_documents(path))


def _required_string(entry: dict[object, object], field: str, index: int) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Knowledge entry {index} {field} must be text")
    return value.strip()
