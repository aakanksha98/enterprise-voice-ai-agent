from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.tools.cancellation import (
    CancellationRequest,
    mock_cancellation_tool,
)


def cancellation_decision(
) -> PlannerDecision:
    return PlannerDecision(
        intent="cancel_appointment",
        confidence=0.97,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )


def create_recording_cancellation_tool(
    calls: list[dict[str, str]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_cancellation", args_schema=CancellationRequest)
    def record_cancellation(service: str, date: str, time: str) -> Any:
        """Record a cancellation request for a deterministic test."""
        calls.append({"service": service, "date": date, "time": time})
        if result is not None:
            return result

        return {
            "service": service,
            "date": date,
            "time": time,
            "status": "cancelled",
        }

    return record_cancellation


def test_mock_cancellation_tool_returns_structured_confirmation() -> None:
    result = mock_cancellation_tool.invoke(
        {
            "service": " dental cleaning ",
            "date": " tomorrow ",
            "time": " 2 PM ",
        }
    )

    assert mock_cancellation_tool.name == "cancel_appointment"
    assert result == {
        "service": "dental cleaning",
        "date": "tomorrow",
        "time": "2 PM",
        "status": "cancelled",
    }


def test_cancellation_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {
            "user_message": "Cancel my appointment",
            "active_appointment": {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        },
        context=context,
    )

    assert tool_calls == [
        {"service": "dental cleaning", "date": "tomorrow", "time": "2 PM"}
    ]
    assert result["workflow_stage"] == "appointment_cancelled"
    assert result["missing_cancellation_slots"] == []
    assert result["cancellation_result"] == {
        "service": "dental cleaning",
        "date": "tomorrow",
        "time": "2 PM",
        "status": "cancelled",
    }
    assert result["active_appointment"] == {
        "service": "dental cleaning",
        "date": "tomorrow",
        "time": "2 PM",
        "status": "cancelled",
    }
    assert result["final_response"] == (
        "Done. I've cancelled your dental cleaning appointment for tomorrow at 2 PM."
    )


def test_cancellation_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool([]),
    )
    updates = agent_graph.stream(
        {
            "user_message": "Cancel my appointment",
            "active_appointment": {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        },
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "cancellation",
        "response",
    ]


def test_cancellation_without_active_appointment_skips_tool() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Cancel my appointment"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "no_active_appointment"
    assert result["missing_cancellation_slots"] == []
    assert result["cancellation_result"] is None
    assert result["final_response"] == (
        "I don't have an active appointment in this conversation to cancel yet."
    )


def test_cancellation_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool(
            [],
            {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        ),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {
                "user_message": "Cancel my appointment",
                "active_appointment": {
                    "service": "dental cleaning",
                    "date": "tomorrow",
                    "time": "2 PM",
                    "status": "confirmed",
                },
            },
            context=context,
        )


def test_complete_cancellation_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
    )

    with pytest.raises(RuntimeError, match="Cancellation tool context is required"):
        agent_graph.invoke(
            {
                "user_message": "Cancel my appointment",
                "active_appointment": {
                    "service": "dental cleaning",
                    "date": "tomorrow",
                    "time": "2 PM",
                    "status": "confirmed",
                },
            },
            context=context,
        )


def test_planned_intent_router_selects_cancellation_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Cancel my appointment",
            "detected_intent": "cancel_appointment",
        }
    ) == "cancellation"
