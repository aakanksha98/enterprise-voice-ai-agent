from typing import Literal

from backend.app.agent.state import AgentState


RouteDecision = Literal["ready", "invalid"]


def route_validated_input(state: AgentState) -> RouteDecision:
    input_status = state.get("input_status")

    if input_status == "valid":
        return "ready"

    if input_status == "invalid":
        return "invalid"

    raise ValueError("Input must be validated before routing")
