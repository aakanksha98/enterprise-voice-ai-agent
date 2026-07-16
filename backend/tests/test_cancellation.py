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
    appointment_id: str | None = "APT-1234ABCD",
) -> PlannerDecision:
    return PlannerDecision(
        intent="cancel_appointment",
        confidence=0.97,
        faq_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            appointment_id=appointment_id,
            escalation_reason=None,
        ),
    )


def create_recording_cancellation_tool(
    calls: list[dict[str, str]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_cancellation", args_schema=CancellationRequest)
    def record_cancellation(appointment_id: str) -> Any:
        """Record a cancellation request for a deterministic test."""
        calls.append({"appointment_id": appointment_id})
        if result is not None:
            return result

        return {"appointment_id": appointment_id, "status": "cancelled"}

    return record_cancellation


def test_mock_cancellation_tool_returns_structured_confirmation() -> None:
    result = mock_cancellation_tool.invoke(
        {"appointment_id": " APT-1234ABCD "}
    )

    assert mock_cancellation_tool.name == "cancel_appointment"
    assert result == {
        "appointment_id": "APT-1234ABCD",
        "status": "cancelled",
    }


def test_cancellation_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Cancel appointment APT-1234ABCD"},
        context=context,
    )

    assert tool_calls == [{"appointment_id": "APT-1234ABCD"}]
    assert result["workflow_stage"] == "appointment_cancelled"
    assert result["missing_cancellation_slots"] == []
    assert result["cancellation_result"] == {
        "appointment_id": "APT-1234ABCD",
        "status": "cancelled",
    }


def test_cancellation_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool([]),
    )
    updates = agent_graph.stream(
        {"user_message": "Cancel appointment APT-1234ABCD"},
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "cancellation",
    ]


def test_missing_appointment_id_skips_cancellation_tool() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision(None)),
        cancellation_tool=create_recording_cancellation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Cancel my appointment"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "cancellation_information_required"
    assert result["missing_cancellation_slots"] == ["appointment_id"]
    assert result["cancellation_result"] is None


def test_cancellation_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
        cancellation_tool=create_recording_cancellation_tool(
            [],
            {"appointment_id": "APT-1234ABCD", "status": "confirmed"},
        ),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {"user_message": "Cancel appointment APT-1234ABCD"},
            context=context,
        )


def test_complete_cancellation_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: cancellation_decision()),
    )

    with pytest.raises(RuntimeError, match="Cancellation tool context is required"):
        agent_graph.invoke(
            {"user_message": "Cancel appointment APT-1234ABCD"},
            context=context,
        )


def test_planned_intent_router_selects_cancellation_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Cancel appointment APT-1234ABCD",
            "detected_intent": "cancel_appointment",
        }
    ) == "cancellation"
