from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict


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
RescheduleSlot = Literal["appointment_id", "date", "time"]
RescheduleStatus = Literal["rescheduled"]
EscalationStatus = Literal["queued"]
ConversationRole = Literal["user", "assistant"]
WorkflowStage = Literal[
    "ready_for_planning",
    "planned",
    "faq_answered",
    "knowledge_retrieved",
    "booking_information_required",
    "appointment_booked",
    "cancellation_information_required",
    "appointment_cancelled",
    "reschedule_information_required",
    "appointment_rescheduled",
    "human_escalation_queued",
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


class RescheduleResult(TypedDict):
    appointment_id: str
    status: RescheduleStatus
    new_date: str
    new_time: str


class EscalationResult(TypedDict):
    escalation_id: str
    status: EscalationStatus
    reason: str | None


class ConversationTurn(TypedDict):
    role: ConversationRole
    content: str


class AgentState(TypedDict):
    user_message: str
    conversation_history: Annotated[list[ConversationTurn], add]
    normalized_message: NotRequired[str]
    input_status: NotRequired[InputStatus]
    workflow_stage: NotRequired[WorkflowStage]
    validation_error: NotRequired[str | None]
    detected_intent: NotRequired[PlannerIntent]
    planner_confidence: NotRequired[float]
    extracted_slots: NotRequired[ExtractedSlots]
    faq_topic: NotRequired[FAQTopic | None]
    draft_response: NotRequired[str | None]
    final_response: NotRequired[str | None]
    retrieval_query: NotRequired[str | None]
    retrieved_documents: NotRequired[list[RetrievedDocument]]
    missing_booking_slots: NotRequired[list[BookingSlot]]
    booking_result: NotRequired[BookingResult | None]
    missing_cancellation_slots: NotRequired[list[CancellationSlot]]
    cancellation_result: NotRequired[CancellationResult | None]
    missing_reschedule_slots: NotRequired[list[RescheduleSlot]]
    reschedule_result: NotRequired[RescheduleResult | None]
    escalation_result: NotRequired[EscalationResult | None]
    previous_workflow_stage: NotRequired[WorkflowStage | None]


class AgentStateUpdate(TypedDict, total=False):
    normalized_message: str
    conversation_history: list[ConversationTurn]
    input_status: InputStatus
    workflow_stage: WorkflowStage
    validation_error: str | None
    detected_intent: PlannerIntent
    planner_confidence: float
    extracted_slots: ExtractedSlots
    faq_topic: FAQTopic | None
    draft_response: str | None
    final_response: str | None
    retrieval_query: str | None
    retrieved_documents: list[RetrievedDocument]
    missing_booking_slots: list[BookingSlot]
    booking_result: BookingResult | None
    missing_cancellation_slots: list[CancellationSlot]
    cancellation_result: CancellationResult | None
    missing_reschedule_slots: list[RescheduleSlot]
    reschedule_result: RescheduleResult | None
    escalation_result: EscalationResult | None
    previous_workflow_stage: WorkflowStage | None
