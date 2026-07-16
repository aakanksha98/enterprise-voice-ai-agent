from typing import Literal, NotRequired, TypedDict


InputStatus = Literal["valid", "invalid"]
WorkflowStage = Literal["ready_for_planning", "rejected"]


class AgentState(TypedDict):
    user_message: str
    normalized_message: NotRequired[str]
    input_status: NotRequired[InputStatus]
    workflow_stage: NotRequired[WorkflowStage]
    validation_error: NotRequired[str]


class AgentStateUpdate(TypedDict, total=False):
    normalized_message: str
    input_status: InputStatus
    workflow_stage: WorkflowStage
    validation_error: str
