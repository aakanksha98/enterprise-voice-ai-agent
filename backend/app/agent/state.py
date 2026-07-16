from typing import Literal, NotRequired, TypedDict


InputStatus = Literal["valid", "invalid"]
FAQTopic = Literal[
    "greeting",
    "assistant_identity",
    "capabilities",
    "courtesy",
]
PlannerIntent = Literal[
    "faq",
    "rag",
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "human_escalation",
    "clarification",
]
WorkflowStage = Literal[
    "ready_for_planning",
    "planned",
    "faq_answered",
    "knowledge_retrieved",
    "rejected",
]


class ExtractedSlots(TypedDict, total=False):
    service: str
    date: str
    time: str
    appointment_id: str
    escalation_reason: str


class RetrievedDocument(TypedDict):
    content: str
    source: str
    category: NotRequired[str]
    similarity: NotRequired[float]


class AgentState(TypedDict):
    user_message: str
    normalized_message: NotRequired[str]
    input_status: NotRequired[InputStatus]
    workflow_stage: NotRequired[WorkflowStage]
    validation_error: NotRequired[str]
    detected_intent: NotRequired[PlannerIntent]
    planner_confidence: NotRequired[float]
    extracted_slots: NotRequired[ExtractedSlots]
    faq_topic: NotRequired[FAQTopic | None]
    draft_response: NotRequired[str]
    retrieval_query: NotRequired[str]
    retrieved_documents: NotRequired[list[RetrievedDocument]]


class AgentStateUpdate(TypedDict, total=False):
    normalized_message: str
    input_status: InputStatus
    workflow_stage: WorkflowStage
    validation_error: str
    detected_intent: PlannerIntent
    planner_confidence: float
    extracted_slots: ExtractedSlots
    faq_topic: FAQTopic | None
    draft_response: str
    retrieval_query: str
    retrieved_documents: list[RetrievedDocument]
