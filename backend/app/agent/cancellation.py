from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    CancellationResult,
)
from backend.app.tools.cancellation import (
    CancellationConfirmation,
    CancellationRequest,
)


def execute_cancellation(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the cancellation intent")

    appointment_id = _clean_appointment_id(
        state.get("extracted_slots", {}).get("appointment_id")
    )
    if not appointment_id:
        return {
            "workflow_stage": "cancellation_information_required",
            "missing_cancellation_slots": ["appointment_id"],
            "cancellation_result": None,
        }

    if runtime.context.cancellation_tool is None:
        raise RuntimeError(
            "Cancellation tool context is required for an appointment ID"
        )

    request = CancellationRequest(appointment_id=appointment_id)
    raw_result = runtime.context.cancellation_tool.invoke(request.model_dump())
    confirmation = CancellationConfirmation.model_validate(raw_result)

    return {
        "workflow_stage": "appointment_cancelled",
        "missing_cancellation_slots": [],
        "cancellation_result": cast(
            CancellationResult,
            confirmation.model_dump(),
        ),
    }


def _clean_appointment_id(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
