from backend.app.agent.state import AgentState, AgentStateUpdate, InputStatus


def validate_input(state: AgentState) -> AgentStateUpdate:
    normalized_message = state["user_message"].strip()
    input_status: InputStatus = "valid" if normalized_message else "invalid"

    return {
        "normalized_message": normalized_message,
        "input_status": input_status,
    }


def mark_ready_for_planning(_: AgentState) -> AgentStateUpdate:
    return {"workflow_stage": "ready_for_planning"}


def reject_invalid_input(_: AgentState) -> AgentStateUpdate:
    return {
        "workflow_stage": "rejected",
        "validation_error": "user_message must not be empty",
    }
