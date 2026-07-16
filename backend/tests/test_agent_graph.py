from collections.abc import Callable

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import AgentContext, PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_validated_input
from backend.app.agent.state import PlannerIntent


def planner_context(
    decision: PlannerDecision,
    on_invoke: Callable[[dict[str, str]], None] | None = None,
) -> AgentContext:
    def plan(planner_input: dict[str, str]) -> PlannerDecision:
        if on_invoke is not None:
            on_invoke(planner_input)
        return decision

    return AgentContext(planner=RunnableLambda(plan))


def test_valid_message_is_normalized_and_planned() -> None:
    planner_inputs: list[dict[str, str]] = []
    decision = PlannerDecision(
        intent="book_appointment",
        confidence=0.94,
        faq_topic=None,
        slots=PlannerSlots(
            service="haircut",
            date="tomorrow",
            time=None,
            appointment_id=None,
            escalation_reason=None,
        ),
    )

    result = agent_graph.invoke(
        {"user_message": "  Book a haircut tomorrow  "},
        context=planner_context(decision, planner_inputs.append),
    )

    assert planner_inputs == [{"user_message": "Book a haircut tomorrow"}]
    assert result == {
        "user_message": "  Book a haircut tomorrow  ",
        "normalized_message": "Book a haircut tomorrow",
        "input_status": "valid",
        "workflow_stage": "planned",
        "detected_intent": "book_appointment",
        "planner_confidence": 0.94,
        "faq_topic": None,
        "extracted_slots": {
            "service": "haircut",
            "date": "tomorrow",
        },
    }


def test_empty_message_routes_to_rejection_without_calling_planner() -> None:
    result = agent_graph.invoke({"user_message": "   "})

    assert result == {
        "user_message": "   ",
        "normalized_message": "",
        "input_status": "invalid",
        "workflow_stage": "rejected",
        "validation_error": "user_message must not be empty",
    }


def test_valid_message_requires_planner_context() -> None:
    with pytest.raises(RuntimeError, match="Planner context is required"):
        agent_graph.invoke({"user_message": "Book an appointment"})


def test_valid_path_runs_planner_after_validation() -> None:
    decision = PlannerDecision(
        intent="rag",
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
    updates = agent_graph.stream(
        {"user_message": "What services do you offer?"},
        context=planner_context(decision),
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
    ]


def test_invalid_path_skips_planner() -> None:
    updates = agent_graph.stream(
        {"user_message": "\t"},
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "reject_invalid_input",
    ]


@pytest.mark.parametrize(
    "intent",
    [
        "faq",
        "rag",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
        "human_escalation",
        "clarification",
    ],
)
def test_planner_schema_accepts_only_supported_intents(
    intent: PlannerIntent,
) -> None:
    decision = PlannerDecision(
        intent=intent,
        confidence=0.8,
        faq_topic="greeting" if intent == "faq" else None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            appointment_id=None,
            escalation_reason=None,
        ),
    )

    assert decision.intent == intent


def test_planner_schema_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError):
        PlannerDecision(
            intent="weather",
            confidence=0.8,
            faq_topic=None,
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                appointment_id=None,
                escalation_reason=None,
            ),
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_planner_schema_rejects_out_of_range_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        PlannerDecision(
            intent="clarification",
            confidence=confidence,
            faq_topic=None,
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                appointment_id=None,
                escalation_reason=None,
            ),
        )


def test_router_rejects_unvalidated_state() -> None:
    with pytest.raises(ValueError, match="must be validated"):
        route_validated_input({"user_message": "Book an appointment"})
