from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import AgentState, AgentStateUpdate, EscalationResult
from backend.app.tools.escalation import (
    HumanEscalationConfirmation,
    HumanEscalationRequest,
)


def execute_human_escalation(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None or runtime.context.escalation_tool is None:
        raise RuntimeError(
            "Human escalation tool context is required for the escalation intent"
        )

    reason = _clean_reason(
        state.get("extracted_slots", {}).get("escalation_reason")
    )
    request = HumanEscalationRequest(reason=reason)
    raw_result = runtime.context.escalation_tool.invoke(request.model_dump())
    confirmation = HumanEscalationConfirmation.model_validate(raw_result)

    return {
        "workflow_stage": "human_escalation_queued",
        "escalation_result": cast(
            EscalationResult,
            confirmation.model_dump(),
        ),
    }


def _clean_reason(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned_value = value.strip()
    return cleaned_value or None
