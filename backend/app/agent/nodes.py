from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.planner import AgentContext
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    ExtractedSlots,
    InputStatus,
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
