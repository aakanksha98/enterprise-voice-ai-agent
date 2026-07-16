from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    ActiveAppointment,
    AgentState,
    AgentStateUpdate,
    BookingResult,
    BookingSlot,
    ExtractedSlots,
)
from backend.app.business_profiles import (
    match_supported_service,
    supported_service_names,
)
from backend.app.tools.booking import BookingConfirmation, BookingRequest


REQUIRED_BOOKING_SLOTS: tuple[BookingSlot, ...] = ("service", "date", "time")
ACTIVE_APPOINTMENT_STATUSES = {"confirmed", "rescheduled"}


def execute_booking(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the booking intent")

    if _current_active_appointment(state) is not None:
        return {
            "workflow_stage": "appointment_already_active",
            "missing_booking_slots": [],
            "booking_result": None,
        }

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

    requested_service = extracted_slots["service"]
    matched_service = match_supported_service(
        runtime.context.business_profile,
        requested_service,
    )
    if matched_service is None:
        return {
            "workflow_stage": "unsupported_service_requested",
            "missing_booking_slots": [],
            "booking_result": None,
            "unsupported_service": requested_service,
            "supported_services": supported_service_names(
                runtime.context.business_profile
            ),
        }

    if runtime.context.booking_tool is None:
        raise RuntimeError(
            "Booking tool context is required for complete booking details"
        )

    canonical_slots = cast(
        ExtractedSlots,
        {**extracted_slots, "service": matched_service},
    )
    request = BookingRequest.model_validate(
        {
            slot: canonical_slots[slot]
            for slot in REQUIRED_BOOKING_SLOTS
        }
    )
    raw_result = runtime.context.booking_tool.invoke(request.model_dump())
    confirmation = BookingConfirmation.model_validate(raw_result)
    appointment = cast(ActiveAppointment, confirmation.model_dump())

    return {
        "workflow_stage": "appointment_booked",
        "extracted_slots": canonical_slots,
        "missing_booking_slots": [],
        "booking_result": cast(BookingResult, confirmation.model_dump()),
        "active_appointment": appointment,
    }


def _clean_slot_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _current_active_appointment(state: AgentState) -> ActiveAppointment | None:
    appointment = state.get("active_appointment")
    if appointment is None:
        return None

    if appointment.get("status") in ACTIVE_APPOINTMENT_STATUSES:
        return appointment

    return None
