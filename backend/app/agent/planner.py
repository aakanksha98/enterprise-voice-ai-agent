import os
from typing import Self

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backend.app.agent.state import FAQTopic, PlannerIntent


DEFAULT_PLANNER_MODEL = "gpt-5.4-mini"


class PlannerSlots(BaseModel):
    """Appointment details stated explicitly in the user's request."""

    model_config = ConfigDict(extra="forbid")

    service: str | None
    date: str | None
    time: str | None
    appointment_id: str | None
    escalation_reason: str | None


class PlannerDecision(BaseModel):
    """A single supported route and any slots found by the planner."""

    model_config = ConfigDict(extra="forbid")

    intent: PlannerIntent
    confidence: float
    faq_topic: FAQTopic | None
    slots: PlannerSlots

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_faq_topic(self) -> Self:
        if self.intent == "faq" and self.faq_topic is None:
            raise ValueError("faq_topic is required for the faq intent")

        if self.intent != "faq" and self.faq_topic is not None:
            raise ValueError("faq_topic is only valid for the faq intent")

        return self


PlannerRunnable = Runnable[dict[str, str], PlannerDecision]


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the constrained intent planner for an enterprise receptionist.

Choose exactly one route:
- faq: greetings, assistant identity, capabilities, or courtesy replies.
- rag: services, pricing, policies, or other business knowledge.
- book_appointment: create a new appointment.
- cancel_appointment: cancel an existing appointment.
- reschedule_appointment: move an existing appointment.
- human_escalation: the user explicitly asks for a person or human handoff.
- clarification: the request is unclear or no single route can be selected.

For faq, set faq_topic to greeting, assistant_identity, capabilities, or
courtesy. For every other route, set faq_topic to null.

Extract only details explicitly present in the current request or its relevant
unfinished prior request. Preserve date and time phrases as spoken. Use prior
conversation only to resolve a direct follow-up. Do not carry details into an
unrelated request. Do not answer the request and do not call any tool. Return a
confidence between 0 and 1 for the route selection.""",
        ),
        (
            "human",
            """Prior conversation:
{conversation_history}

Classify the current request:
{user_message}""",
        ),
    ]
)


def create_openai_planner(model: str | None = None) -> PlannerRunnable:
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_PLANNER_MODEL)
    chat_model = ChatOpenAI(model=selected_model, max_retries=2)
    structured_model = chat_model.with_structured_output(
        PlannerDecision,
        method="json_schema",
        strict=True,
    )
    return PLANNER_PROMPT | structured_model
