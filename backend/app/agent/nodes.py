from typing import cast

from langchain_core.documents import Document
from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    ExtractedSlots,
    InputStatus,
    RetrievedDocument,
)


def validate_input(state: AgentState) -> AgentStateUpdate:
    normalized_message = state["user_message"].strip()
    input_status: InputStatus = "valid" if normalized_message else "invalid"

    return {
        "normalized_message": normalized_message,
        "input_status": input_status,
    }


def mark_ready_for_planning(_: AgentState) -> AgentStateUpdate:
    return {"workflow_stage": "ready_for_planning"}


def reject_invalid_input(_: AgentState) -> AgentStateUpdate:
    return {
        "workflow_stage": "rejected",
        "validation_error": "user_message must not be empty",
    }


def plan_request(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Planner context is required for valid input")

    normalized_message = state.get("normalized_message")
    if not normalized_message:
        raise ValueError("A normalized message is required before planning")

    decision = runtime.context.planner.invoke(
        {"user_message": normalized_message}
    )
    extracted_slots = cast(
        ExtractedSlots,
        decision.slots.model_dump(exclude_none=True),
    )

    return {
        "workflow_stage": "planned",
        "detected_intent": decision.intent,
        "planner_confidence": decision.confidence,
        "extracted_slots": extracted_slots,
        "faq_topic": decision.faq_topic,
    }


def retrieve_business_knowledge(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None or runtime.context.rag_retriever is None:
        raise RuntimeError("RAG retriever context is required for the rag intent")

    normalized_message = state.get("normalized_message")
    if not normalized_message:
        raise ValueError("A normalized message is required before retrieval")

    documents = runtime.context.rag_retriever.invoke(normalized_message)
    retrieved_documents = [
        serialized
        for document in documents
        if (serialized := _serialize_document(document)) is not None
    ]

    return {
        "workflow_stage": "knowledge_retrieved",
        "retrieval_query": normalized_message,
        "retrieved_documents": retrieved_documents,
    }


def _serialize_document(document: Document) -> RetrievedDocument | None:
    content = document.page_content.strip()
    if not content:
        return None

    metadata = document.metadata
    source = str(metadata.get("source") or "business_knowledge")
    retrieved_document: RetrievedDocument = {
        "content": content,
        "source": source,
    }

    category = metadata.get("category")
    if category:
        retrieved_document["category"] = str(category)

    similarity = metadata.get("similarity")
    if isinstance(similarity, (int, float)) and not isinstance(similarity, bool):
        retrieved_document["similarity"] = float(similarity)

    return retrieved_document
