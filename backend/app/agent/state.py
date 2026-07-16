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
BookingSlot = Literal["service", "date", "time"]
BookingStatus = Literal["confirmed"]
CancellationSlot = Literal["appointment_id"]
CancellationStatus = Literal["cancelled"]
WorkflowStage = Literal[
    "ready_for_planning",
    "planned",
    "faq_answered",
    "knowledge_retrieved",
    "booking_information_required",
    "appointment_booked",
    "cancellation_information_required",
    "appointment_cancelled",
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


class BookingResult(TypedDict):
    appointment_id: str
    status: BookingStatus
    service: str
    date: str
    time: str


class CancellationResult(TypedDict):
    appointment_id: str
    status: CancellationStatus


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
    missing_booking_slots: NotRequired[list[BookingSlot]]
    booking_result: NotRequired[BookingResult | None]
    missing_cancellation_slots: NotRequired[list[CancellationSlot]]
    cancellation_result: NotRequired[CancellationResult | None]


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
    missing_booking_slots: list[BookingSlot]
    booking_result: BookingResult | None
    missing_cancellation_slots: list[CancellationSlot]
    cancellation_result: CancellationResult | None
