from re import search
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
    BusinessProfile,
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
    schedule_violation = _schedule_violation(
        runtime.context.business_profile,
        _clean_slot_value(extracted_slots.get("date")),
        _clean_slot_value(extracted_slots.get("time")),
    )
    if schedule_violation is not None:
        return {
            "workflow_stage": "appointment_outside_business_hours",
            "missing_booking_slots": [],
            "booking_result": None,
            "schedule_violation": schedule_violation,
            "business_hours": runtime.context.business_profile.booking_hours.description,
        }

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


WEEKDAY_INDEXES = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


def _schedule_violation(
    profile: BusinessProfile,
    requested_date: str,
    requested_time: str,
) -> str | None:
    weekday = _parse_weekday(requested_date)
    time_minutes = _parse_time_minutes(requested_time)
    if weekday is None or time_minutes is None:
        return None

    for window_weekday, start_minutes, end_minutes in profile.booking_hours.weekly_windows:
        if window_weekday == weekday and start_minutes <= time_minutes < end_minutes:
            return None

    return (
        f"{requested_date} at {requested_time} is outside business hours. "
        f"Available booking hours are {profile.booking_hours.description}."
    )


def _parse_weekday(value: str) -> int | None:
    normalized = value.strip().lower()
    for token, weekday in WEEKDAY_INDEXES.items():
        if search(rf"\b{token}\b", normalized):
            return weekday

    return None


def _parse_time_minutes(value: str) -> int | None:
    match = search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", value.strip().lower())
    if match is None:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    meridiem = match.group(3)
    if hour < 1 or hour > 12 or minute > 59:
        return None

    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0

    return hour * 60 + minute
