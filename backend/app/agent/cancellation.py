from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    ActiveAppointment,
    AgentState,
    AgentStateUpdate,
    CancellationResult,
)
from backend.app.tools.cancellation import (
    CancellationConfirmation,
    CancellationRequest,
)


ACTIVE_APPOINTMENT_STATUSES = {"confirmed", "rescheduled"}


def execute_cancellation(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the cancellation intent")

    appointment = _current_active_appointment(state)
    if appointment is None:
        return {
            "workflow_stage": "no_active_appointment",
            "missing_cancellation_slots": [],
            "cancellation_result": None,
        }

    if runtime.context.cancellation_tool is None:
        raise RuntimeError(
            "Cancellation tool context is required for an active appointment"
        )

    request = CancellationRequest(
        service=appointment["service"],
        date=appointment["date"],
        time=appointment["time"],
    )
    raw_result = runtime.context.cancellation_tool.invoke(request.model_dump())
    confirmation = CancellationConfirmation.model_validate(raw_result)
    cancelled_appointment = cast(ActiveAppointment, confirmation.model_dump())

    return {
        "workflow_stage": "appointment_cancelled",
        "missing_cancellation_slots": [],
        "cancellation_result": cast(
            CancellationResult,
            confirmation.model_dump(),
        ),
        "active_appointment": cancelled_appointment,
    }


def _current_active_appointment(state: AgentState) -> ActiveAppointment | None:
    appointment = state.get("active_appointment")
    if appointment is None:
        return None

    if appointment.get("status") in ACTIVE_APPOINTMENT_STATUSES:
        return appointment

    return None
