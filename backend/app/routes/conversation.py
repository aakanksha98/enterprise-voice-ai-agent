from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent.state import PlannerIntent, WorkflowStage
from backend.app.services.conversation import (
    ConversationService,
    get_conversation_service,
)


class ConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=500)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")


class ConversationResponse(BaseModel):
    response: str
    intent: PlannerIntent | None
    workflow_stage: WorkflowStage
    slots: dict[str, str]
    missing_fields: list[str]
    reference_id: str | None


router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.post(
    "",
    response_model=ConversationResponse,
    summary="Continue a voice-agent conversation",
)
def continue_conversation(
    request: ConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    state = service.respond(request.message, request.session_id)
    return ConversationResponse(
        response=state["final_response"] or "",
        intent=state.get("detected_intent"),
        workflow_stage=state["workflow_stage"],
        slots=dict(state.get("extracted_slots", {})),
        missing_fields=_missing_fields(state),
        reference_id=_reference_id(state),
    )


def _missing_fields(state: dict[str, Any]) -> list[str]:
    for field_name in (
        "missing_booking_slots",
        "missing_cancellation_slots",
        "missing_reschedule_slots",
    ):
        fields = state.get(field_name)
        if fields:
            return list(fields)
    return []


def _reference_id(state: dict[str, Any]) -> str | None:
    result_fields = (
        ("booking_result", "appointment_id"),
        ("cancellation_result", "appointment_id"),
        ("reschedule_result", "appointment_id"),
        ("escalation_result", "escalation_id"),
    )
    for result_name, identifier_name in result_fields:
        result = state.get(result_name)
        if isinstance(result, dict) and result.get(identifier_name):
            return str(result[identifier_name])
    return None
