import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.faq import FAQ_RESPONSES, answer_faq
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.agent.state import FAQTopic


@pytest.mark.parametrize(("faq_topic", "expected_response"), FAQ_RESPONSES.items())
def test_faq_catalog_returns_stable_draft_response(
    faq_topic: FAQTopic,
    expected_response: str,
) -> None:
    result = answer_faq(
        {
            "user_message": "FAQ request",
            "faq_topic": faq_topic,
        }
    )

    assert result == {
        "workflow_stage": "faq_answered",
        "draft_response": expected_response,
    }


def test_faq_plan_routes_through_faq_node() -> None:
    decision = PlannerDecision(
        intent="faq",
        confidence=0.98,
        faq_topic="greeting",
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            appointment_id=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(planner=RunnableLambda(lambda _: decision))
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
        "faq",
        "response",
    ]
    assert updates[-2]["faq"] == {
        "workflow_stage": "faq_answered",
        "draft_response": FAQ_RESPONSES["greeting"],
    }
    assert updates[-1]["response"]["final_response"] == FAQ_RESPONSES["greeting"]


def test_faq_node_rejects_missing_topic() -> None:
    with pytest.raises(ValueError, match="FAQ topic is required"):
        answer_faq({"user_message": "Hello"})


def test_planned_intent_router_requires_planner_output() -> None:
    with pytest.raises(ValueError, match="planned intent is required"):
        route_planned_intent({"user_message": "Hello"})


def test_planner_schema_requires_topic_for_faq_intent() -> None:
    with pytest.raises(ValidationError, match="faq_topic is required"):
        PlannerDecision(
            intent="faq",
            confidence=0.9,
            faq_topic=None,
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                appointment_id=None,
                escalation_reason=None,
            ),
        )


def test_planner_schema_rejects_faq_topic_for_other_intents() -> None:
    with pytest.raises(ValidationError, match="only valid for the faq intent"):
        PlannerDecision(
            intent="rag",
            confidence=0.9,
            faq_topic="greeting",
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                appointment_id=None,
                escalation_reason=None,
            ),
        )
