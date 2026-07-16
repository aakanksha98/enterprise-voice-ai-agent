import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.response import SMALL_TALK_RESPONSES
from backend.app.agent.routing import route_planned_intent
from backend.app.agent.state import SmallTalkTopic


def _small_talk_decision(topic: SmallTalkTopic) -> PlannerDecision:
    return PlannerDecision(
        intent="small_talk",
        confidence=0.98,
        small_talk_topic=topic,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            appointment_id=None,
            escalation_reason=None,
        ),
    )


@pytest.mark.parametrize(
    ("small_talk_topic", "expected_response"),
    SMALL_TALK_RESPONSES.items(),
)
def test_small_talk_catalog_returns_stable_final_response(
    small_talk_topic: SmallTalkTopic,
    expected_response: str,
) -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: _small_talk_decision(small_talk_topic))
    )

    result = agent_graph.invoke(
        {"user_message": "Small-talk request"},
        context=context,
    )

    assert result["workflow_stage"] == "planned"
    assert result["small_talk_topic"] == small_talk_topic
    assert result["final_response"] == expected_response


def test_small_talk_plan_routes_directly_to_response() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: _small_talk_decision("greeting"))
    )
    updates = list(
        agent_graph.stream(
            {"user_message": "Hello"},
            context=context,
            stream_mode="updates",
        )
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "response",
    ]
    assert updates[-1]["response"]["final_response"] == (
        SMALL_TALK_RESPONSES["greeting"]
    )


def test_planned_intent_router_requires_planner_output() -> None:
    with pytest.raises(ValueError, match="planned intent is required"):
        route_planned_intent({"user_message": "Hello"})


def test_planner_schema_requires_topic_for_small_talk_intent() -> None:
    with pytest.raises(ValidationError, match="small_talk_topic is required"):
        PlannerDecision(
            intent="small_talk",
            confidence=0.9,
            small_talk_topic=None,
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                appointment_id=None,
                escalation_reason=None,
            ),
        )


def test_planner_schema_rejects_small_talk_topic_for_other_intents() -> None:
    with pytest.raises(
        ValidationError,
        match="only valid for the small_talk intent",
    ):
        PlannerDecision(
            intent="rag",
            confidence=0.9,
            small_talk_topic="greeting",
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                appointment_id=None,
                escalation_reason=None,
            ),
        )
