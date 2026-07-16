from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.tools.booking import BookingRequest, mock_booking_tool


def booking_decision(
    *,
    service: str | None = "dental cleaning",
    date: str | None = "tomorrow",
    time: str | None = "2 PM",
) -> PlannerDecision:
    return PlannerDecision(
        intent="book_appointment",
        confidence=0.96,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=service,
            date=date,
            time=time,
            escalation_reason=None,
        ),
    )


def create_recording_booking_tool(
    calls: list[dict[str, str]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_booking", args_schema=BookingRequest)
    def record_booking(
        service: str,
        date: str,
        time: str,
    ) -> Any:
        """Record a booking request for a deterministic test."""
        calls.append({"service": service, "date": date, "time": time})
        if result is not None:
            return result

        return {
            "status": "confirmed",
            "service": service,
            "date": date,
            "time": time,
        }

    return record_booking


def test_mock_booking_tool_returns_structured_confirmation() -> None:
    result = mock_booking_tool.invoke(
        {"service": " dental cleaning ", "date": " tomorrow ", "time": " 2 PM "}
    )

    assert mock_booking_tool.name == "book_appointment"
    assert result == {
        "service": "dental cleaning",
        "date": "tomorrow",
        "time": "2 PM",
        "status": "confirmed",
    }


def test_booking_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning tomorrow at 2 PM"},
        context=context,
    )

    assert tool_calls == [
        {"service": "dental cleaning", "date": "tomorrow", "time": "2 PM"}
    ]
    assert result == {
        "user_message": "Book a dental cleaning tomorrow at 2 PM",
        "conversation_history": [
            {
                "role": "user",
                "content": "Book a dental cleaning tomorrow at 2 PM",
            },
            {
                "role": "assistant",
                "content": (
                    "Your dental cleaning appointment is booked for tomorrow at 2 PM."
                ),
            },
        ],
        "normalized_message": "Book a dental cleaning tomorrow at 2 PM",
        "input_status": "valid",
        "workflow_stage": "appointment_booked",
        "detected_intent": "book_appointment",
        "planner_confidence": 0.96,
        "extracted_slots": {
            "service": "dental cleaning",
            "date": "tomorrow",
            "time": "2 PM",
        },
        "small_talk_topic": None,
        "missing_booking_slots": [],
        "booking_result": {
            "service": "dental cleaning",
            "date": "tomorrow",
            "time": "2 PM",
            "status": "confirmed",
        },
        "active_appointment": {
            "service": "dental cleaning",
            "date": "tomorrow",
            "time": "2 PM",
            "status": "confirmed",
        },
        "final_response": (
            "Your dental cleaning appointment is booked for tomorrow at 2 PM."
        ),
    }


def test_booking_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
        booking_tool=create_recording_booking_tool([]),
    )
    updates = agent_graph.stream(
        {"user_message": "Book a dental cleaning tomorrow at 2 PM"},
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "booking",
        "response",
    ]


def test_missing_booking_slots_skip_tool_invocation() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision(time=None)),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning tomorrow"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "booking_information_required"
    assert result["missing_booking_slots"] == ["time"]
    assert result["booking_result"] is None
    assert result["final_response"] == (
        "To book the appointment, please provide time."
    )


def test_booking_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
        booking_tool=create_recording_booking_tool([], {"status": "confirmed"}),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {"user_message": "Book a dental cleaning tomorrow at 2 PM"},
            context=context,
        )


def test_complete_booking_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
    )

    with pytest.raises(RuntimeError, match="Booking tool context is required"):
        agent_graph.invoke(
            {"user_message": "Book a dental cleaning tomorrow at 2 PM"},
            context=context,
        )


def test_planned_intent_router_selects_booking_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Book a dental cleaning tomorrow at 2 PM",
            "detected_intent": "book_appointment",
        }
    ) == "booking"


def test_unsupported_service_skips_booking_tool() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision(service="oil change")),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book an oil change tomorrow at 2 PM"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "unsupported_service_requested"
    assert result["unsupported_service"] == "oil change"
    assert result["supported_services"] == [
        "dental cleaning",
        "dental exam",
        "teeth whitening",
        "filling",
        "emergency dental visit",
    ]
    assert result["final_response"] == (
        "I can't book oil change for this demo profile. I can help with "
        "dental cleaning, dental exam, teeth whitening, filling, and "
        "emergency dental visit. Which service would you like?"
    )
