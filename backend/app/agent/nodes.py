from typing import cast

from langchain_core.documents import Document
from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    ExtractedSlots,
    InputStatus,
    PlannerIntent,
    RetrievedDocument,
    WorkflowStage,
)


FOLLOW_UP_STAGES: dict[PlannerIntent, WorkflowStage] = {
    "book_appointment": "booking_information_required",
    "cancel_appointment": "cancellation_information_required",
    "reschedule_appointment": "reschedule_information_required",
}


def validate_input(state: AgentState) -> AgentStateUpdate:
    normalized_message = state["user_message"].strip()
    input_status: InputStatus = "valid" if normalized_message else "invalid"

    return {
        "normalized_message": normalized_message,
        "input_status": input_status,
    }


def mark_ready_for_planning(state: AgentState) -> AgentStateUpdate:
    normalized_message = state.get("normalized_message")
    if not normalized_message:
        raise ValueError("A normalized message is required before planning")

    update: AgentStateUpdate = {
        "workflow_stage": "ready_for_planning",
        "conversation_history": [
            {"role": "user", "content": normalized_message}
        ],
    }
    if state.get("conversation_history"):
        previous_stage = state.get("workflow_stage")
        if previous_stage == "rejected":
            previous_stage = state.get("previous_workflow_stage")

        update.update(_reset_turn_outputs())
        update["previous_workflow_stage"] = previous_stage

    return update


def reject_invalid_input(state: AgentState) -> AgentStateUpdate:
    update: AgentStateUpdate = {}
    if state.get("conversation_history"):
        update.update(_reset_turn_outputs())
        update["previous_workflow_stage"] = state.get("workflow_stage")

    update.update(
        {
            "workflow_stage": "rejected",
            "validation_error": "user_message must not be empty",
        }
    )
    return update


def _reset_turn_outputs() -> AgentStateUpdate:
    return {
        "validation_error": None,
        "draft_response": None,
        "final_response": None,
        "retrieval_query": None,
        "retrieved_documents": [],
        "missing_booking_slots": [],
        "booking_result": None,
        "missing_cancellation_slots": [],
        "cancellation_result": None,
        "missing_reschedule_slots": [],
        "reschedule_result": None,
        "escalation_result": None,
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
        {
            "user_message": normalized_message,
            "conversation_history": _format_prior_conversation(state),
        }
    )
    current_slots = cast(
        ExtractedSlots,
        decision.slots.model_dump(exclude_none=True),
    )
    extracted_slots = _merge_follow_up_slots(state, decision.intent, current_slots)

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


def _format_prior_conversation(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    prior_turns = history[:-1]
    if not prior_turns:
        return "No prior conversation."

    return "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in prior_turns
    )


def _merge_follow_up_slots(
    state: AgentState,
    current_intent: PlannerIntent,
    current_slots: ExtractedSlots,
) -> ExtractedSlots:
    expected_stage = FOLLOW_UP_STAGES.get(current_intent)
    if (
        expected_stage is None
        or state.get("detected_intent") != current_intent
        or state.get("previous_workflow_stage") != expected_stage
    ):
        return current_slots

    return cast(
        ExtractedSlots,
        {**state.get("extracted_slots", {}), **current_slots},
    )
