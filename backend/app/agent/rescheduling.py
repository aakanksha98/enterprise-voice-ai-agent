from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    RescheduleResult,
    RescheduleSlot,
)
from backend.app.tools.rescheduling import (
    RescheduleConfirmation,
    RescheduleRequest,
)


REQUIRED_RESCHEDULE_SLOTS: tuple[RescheduleSlot, ...] = (
    "appointment_id",
    "date",
    "time",
)


def execute_reschedule(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the reschedule intent")

    extracted_slots = state.get("extracted_slots", {})
    missing_slots = [
        slot
        for slot in REQUIRED_RESCHEDULE_SLOTS
        if not _clean_slot_value(extracted_slots.get(slot))
    ]
    if missing_slots:
        return {
            "workflow_stage": "reschedule_information_required",
            "missing_reschedule_slots": missing_slots,
            "reschedule_result": None,
        }

    if runtime.context.reschedule_tool is None:
        raise RuntimeError(
            "Reschedule tool context is required for complete reschedule details"
        )

    request = RescheduleRequest(
        appointment_id=extracted_slots["appointment_id"],
        new_date=extracted_slots["date"],
        new_time=extracted_slots["time"],
    )
    raw_result = runtime.context.reschedule_tool.invoke(request.model_dump())
    confirmation = RescheduleConfirmation.model_validate(raw_result)

    return {
        "workflow_stage": "appointment_rescheduled",
        "missing_reschedule_slots": [],
        "reschedule_result": cast(
            RescheduleResult,
            confirmation.model_dump(),
        ),
    }


def _clean_slot_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
