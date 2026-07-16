from typing import Literal

from backend.app.agent.state import AgentState


RouteDecision = Literal["ready", "invalid"]
PlannedRouteDecision = Literal[
    "faq",
    "rag",
    "booking",
    "cancellation",
    "deferred",
]


def route_validated_input(state: AgentState) -> RouteDecision:
    input_status = state.get("input_status")

    if input_status == "valid":
        return "ready"

    if input_status == "invalid":
        return "invalid"

    raise ValueError("Input must be validated before routing")


def route_planned_intent(state: AgentState) -> PlannedRouteDecision:
    detected_intent = state.get("detected_intent")
    if detected_intent is None:
        raise ValueError("A planned intent is required before routing")

    if detected_intent == "faq":
        return "faq"

    if detected_intent == "rag":
        return "rag"

    if detected_intent == "book_appointment":
        return "booking"

    if detected_intent == "cancel_appointment":
        return "cancellation"

    return "deferred"
