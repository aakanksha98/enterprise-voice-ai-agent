from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.tools.rescheduling import (
    RescheduleRequest,
    mock_reschedule_tool,
)


def reschedule_decision(
    *,
    appointment_id: str | None = "APT-1234ABCD",
    date: str | None = "Friday",
    time: str | None = "4 PM",
) -> PlannerDecision:
    return PlannerDecision(
        intent="reschedule_appointment",
        confidence=0.95,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=date,
            time=time,
            appointment_id=appointment_id,
            escalation_reason=None,
        ),
    )


def create_recording_reschedule_tool(
    calls: list[dict[str, str]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_reschedule", args_schema=RescheduleRequest)
    def record_reschedule(
        appointment_id: str,
        new_date: str,
        new_time: str,
    ) -> Any:
        """Record a reschedule request for a deterministic test."""
        calls.append(
            {
                "appointment_id": appointment_id,
                "new_date": new_date,
                "new_time": new_time,
            }
        )
        if result is not None:
            return result

        return {
            "appointment_id": appointment_id,
            "status": "rescheduled",
            "new_date": new_date,
            "new_time": new_time,
        }

    return record_reschedule


def test_mock_reschedule_tool_returns_structured_confirmation() -> None:
    result = mock_reschedule_tool.invoke(
        {
            "appointment_id": " APT-1234ABCD ",
            "new_date": " Friday ",
            "new_time": " 4 PM ",
        }
    )

    assert mock_reschedule_tool.name == "reschedule_appointment"
    assert result == {
        "appointment_id": "APT-1234ABCD",
        "new_date": "Friday",
        "new_time": "4 PM",
        "status": "rescheduled",
    }


def test_reschedule_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Move appointment APT-1234ABCD to Friday at 4 PM"},
        context=context,
    )

    assert tool_calls == [
        {
            "appointment_id": "APT-1234ABCD",
            "new_date": "Friday",
            "new_time": "4 PM",
        }
    ]
    assert result["workflow_stage"] == "appointment_rescheduled"
    assert result["missing_reschedule_slots"] == []
    assert result["reschedule_result"] == {
        "appointment_id": "APT-1234ABCD",
        "new_date": "Friday",
        "new_time": "4 PM",
        "status": "rescheduled",
    }
    assert result["final_response"] == (
        "Appointment APT-1234ABCD has been rescheduled to Friday at 4 PM."
    )


def test_reschedule_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool([]),
    )
    updates = agent_graph.stream(
        {"user_message": "Move appointment APT-1234ABCD to Friday at 4 PM"},
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "reschedule",
        "response",
    ]


def test_missing_reschedule_slots_skip_tool_invocation() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: reschedule_decision(date=None, time=None)
        ),
        reschedule_tool=create_recording_reschedule_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Reschedule appointment APT-1234ABCD"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "reschedule_information_required"
    assert result["missing_reschedule_slots"] == ["date", "time"]
    assert result["reschedule_result"] is None
    assert result["final_response"] == (
        "To reschedule the appointment, please provide date and time."
    )


def test_reschedule_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool(
            [],
            {
                "appointment_id": "APT-1234ABCD",
                "new_date": "Friday",
                "new_time": "4 PM",
                "status": "confirmed",
            },
        ),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {"user_message": "Move appointment APT-1234ABCD to Friday at 4 PM"},
            context=context,
        )


def test_complete_reschedule_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
    )

    with pytest.raises(RuntimeError, match="Reschedule tool context is required"):
        agent_graph.invoke(
            {"user_message": "Move appointment APT-1234ABCD to Friday at 4 PM"},
            context=context,
        )


def test_planned_intent_router_selects_reschedule_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Move appointment APT-1234ABCD to Friday at 4 PM",
            "detected_intent": "reschedule_appointment",
        }
    ) == "reschedule"
