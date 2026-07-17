from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from backend.app.agent.state import PlannerIntent, WorkflowStage
from backend.app.business_profiles import (
    BUSINESS_NAME_MAX_LENGTH,
    BusinessType,
    sanitize_business_name,
)
from backend.app.services.conversation import (
    ConversationService,
    get_conversation_service,
)


class ConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=500)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    business_type: BusinessType
    business_name: str = Field(
        min_length=1,
        max_length=BUSINESS_NAME_MAX_LENGTH,
    )


class ConversationResponse(BaseModel):
    response: str
    business_type: BusinessType
    business_name: str
    intent: PlannerIntent | None
    workflow_stage: WorkflowStage
    slots: dict[str, str]
    missing_fields: list[str]
    active_appointment: dict[str, str] | None


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
    state = service.respond(
        request.message,
        request.session_id,
        business_type=request.business_type,
        business_name=request.business_name,
    )
    return ConversationResponse(
        response=state["final_response"] or "",
        business_type=request.business_type,
        business_name=sanitize_business_name(request.business_name),
        intent=state.get("detected_intent"),
        workflow_stage=state["workflow_stage"],
        slots=dict(state.get("extracted_slots", {})),
        missing_fields=_missing_fields(state),
        active_appointment=_active_appointment(state),
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


def _active_appointment(state: dict[str, Any]) -> dict[str, str] | None:
    appointment = state.get("active_appointment")
    if not isinstance(appointment, dict):
        return None

    required_fields = ("service", "date", "time", "status")
    if not all(isinstance(appointment.get(field), str) for field in required_fields):
        return None

    return {field: appointment[field] for field in required_fields}
