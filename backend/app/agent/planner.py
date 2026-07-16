import os
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, field_validator

from backend.app.agent.state import PlannerIntent


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
    slots: PlannerSlots

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value


PlannerRunnable = Runnable[dict[str, str], PlannerDecision]


@dataclass(frozen=True)
class AgentContext:
    planner: PlannerRunnable


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the constrained intent planner for an enterprise receptionist.

Choose exactly one route:
- faq: greetings or a very small, stable canned answer.
- rag: services, pricing, policies, or other business knowledge.
- book_appointment: create a new appointment.
- cancel_appointment: cancel an existing appointment.
- reschedule_appointment: move an existing appointment.
- human_escalation: the user explicitly asks for a person or human handoff.
- clarification: the request is unclear or no single route can be selected.

Extract only details explicitly present in the request. Preserve date and time
phrases as spoken. Do not answer the request and do not call any tool. Return a
confidence between 0 and 1 for the route selection.""",
        ),
        ("human", "Classify this request:\n{user_message}"),
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
