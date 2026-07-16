from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    BookingResult,
    BookingSlot,
)
from backend.app.tools.booking import BookingConfirmation, BookingRequest


REQUIRED_BOOKING_SLOTS: tuple[BookingSlot, ...] = ("service", "date", "time")


def execute_booking(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the booking intent")

    extracted_slots = state.get("extracted_slots", {})
    missing_slots = [
        slot
        for slot in REQUIRED_BOOKING_SLOTS
        if not _clean_slot_value(extracted_slots.get(slot))
    ]
    if missing_slots:
        return {
            "workflow_stage": "booking_information_required",
            "missing_booking_slots": missing_slots,
            "booking_result": None,
        }

    if runtime.context.booking_tool is None:
        raise RuntimeError(
            "Booking tool context is required for complete booking details"
        )

    request = BookingRequest.model_validate(
        {
            slot: extracted_slots[slot]
            for slot in REQUIRED_BOOKING_SLOTS
        }
    )
    raw_result = runtime.context.booking_tool.invoke(request.model_dump())
    confirmation = BookingConfirmation.model_validate(raw_result)

    return {
        "workflow_stage": "appointment_booked",
        "missing_booking_slots": [],
        "booking_result": cast(BookingResult, confirmation.model_dump()),
    }


def _clean_slot_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
