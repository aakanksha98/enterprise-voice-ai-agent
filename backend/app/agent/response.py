from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from backend.app.agent.state import AgentState, AgentStateUpdate, SmallTalkTopic


if TYPE_CHECKING:
    from backend.app.agent.context import AgentContext


DEFAULT_RESPONSE_MODEL = "gpt-5.4-mini"
ResponseRunnable = Runnable[dict[str, str], str]


SMALL_TALK_RESPONSES: Final[dict[SmallTalkTopic, str]] = {
    "greeting": (
        "Hi! I'm Aster, the AI receptionist for this business. "
        "How can I help today?"
    ),
    "assistant_identity": "I'm Aster, an AI reception assistant.",
    "capabilities": (
        "I can help with business questions, appointment booking, cancellation, "
        "rescheduling, or connecting you with a person."
    ),
    "courtesy": "You're welcome. Is there anything else I can help with?",
}


RAG_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You answer business questions for a voice receptionist.

Use only the supplied business evidence. If the evidence does not answer the
question, say that you do not have enough information. Never invent prices,
policies, services, availability, or contact details. Keep the answer concise,
natural when spoken aloud, and free of markdown.""",
        ),
        (
            "human",
            """Business evidence:
{retrieved_context}

Customer question:
{user_message}""",
        ),
    ]
)


def create_openai_response_generator(
    model: str | None = None,
) -> ResponseRunnable:
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_RESPONSE_MODEL)
    chat_model = ChatOpenAI(model=selected_model, max_retries=2)
    return RAG_RESPONSE_PROMPT | chat_model | StrOutputParser()


def generate_response(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    response_text = _select_response(state, runtime).strip()
    if not response_text:
        raise ValueError("Response generation returned empty text")

    update: AgentStateUpdate = {"final_response": response_text}
    if state.get("input_status") == "valid":
        update["conversation_history"] = [
            {"role": "assistant", "content": response_text}
        ]
    return update


def _select_response(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> str:
    workflow_stage = state.get("workflow_stage")
    response_builders: dict[str, Callable[[AgentState], str]] = {
        "rejected": _invalid_input_response,
        "booking_information_required": _booking_information_response,
        "appointment_booked": _booking_confirmation_response,
        "appointment_already_active": _appointment_already_active_response,
        "unsupported_service_requested": _unsupported_service_response,
        "cancellation_information_required": _cancellation_information_response,
        "appointment_cancelled": _cancellation_confirmation_response,
        "reschedule_information_required": _reschedule_information_response,
        "appointment_rescheduled": _reschedule_confirmation_response,
        "no_active_appointment": _no_active_appointment_response,
        "human_escalation_queued": _escalation_response,
    }

    if workflow_stage == "knowledge_retrieved":
        return _rag_response(state, runtime)

    if workflow_stage == "planned":
        return _planned_response(state, runtime)

    builder = response_builders.get(workflow_stage or "")
    if builder is None:
        raise ValueError(f"No response strategy for workflow stage: {workflow_stage}")
    return builder(state)


def _rag_response(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> str:
    documents = state.get("retrieved_documents", [])
    if not documents:
        return "I could not find enough business information to answer that."

    if runtime.context is None or runtime.context.response_generator is None:
        raise RuntimeError("Response generator context is required for RAG answers")

    context_blocks = [
        f"Source: {document['source']}\n{document['content']}"
        for document in documents
    ]
    return runtime.context.response_generator.invoke(
        {
            "user_message": state.get("normalized_message", ""),
            "retrieved_context": "\n\n".join(context_blocks),
        }
    )


def _invalid_input_response(_: AgentState) -> str:
    return "Please say or enter a request so I can help."


def _planned_response(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> str:
    if state.get("detected_intent") == "clarification":
        return _clarification_response(state)

    if state.get("detected_intent") != "small_talk":
        raise ValueError("Planned response requires small talk or clarification")

    topic = state.get("small_talk_topic")
    if topic is None:
        raise ValueError("Small-talk topic is required")

    if topic == "greeting":
        return (
            "Hi! I'm Aster, the AI receptionist for "
            f"{_business_name(runtime)}. How can I help today?"
        )

    return SMALL_TALK_RESPONSES[topic]


def _booking_information_response(state: AgentState) -> str:
    missing_fields = _join_fields(state.get("missing_booking_slots", []))
    return f"To book the appointment, please provide {missing_fields}."


def _booking_confirmation_response(state: AgentState) -> str:
    result = state.get("booking_result")
    if result is None:
        raise ValueError("Booking result is required")
    return (
        f"Your {result['service']} appointment is booked for {result['date']} "
        f"at {result['time']}."
    )


def _appointment_already_active_response(state: AgentState) -> str:
    appointment = state.get("active_appointment")
    if appointment is None:
        raise ValueError("Active appointment is required")
    return (
        f"You already have an active {appointment['service']} appointment for "
        f"{appointment['date']} at {appointment['time']}. In this V1 demo, "
        "please cancel or reschedule it before booking another appointment."
    )


def _unsupported_service_response(state: AgentState) -> str:
    requested_service = state.get("unsupported_service") or "that service"
    services = state.get("supported_services", [])
    if not services:
        raise ValueError("Supported services are required")

    return (
        f"I can't book {requested_service} for this demo profile. "
        f"I can help with {_join_fields(services)}. Which service would you like?"
    )


def _cancellation_information_response(_: AgentState) -> str:
    return "I need an active appointment in this conversation before I can cancel it."


def _cancellation_confirmation_response(state: AgentState) -> str:
    result = state.get("cancellation_result")
    if result is None:
        raise ValueError("Cancellation result is required")
    return (
        f"Done. I've cancelled your {result['service']} appointment for "
        f"{result['date']} at {result['time']}."
    )


def _reschedule_information_response(state: AgentState) -> str:
    missing_fields = _join_fields(state.get("missing_reschedule_slots", []))
    return f"To reschedule the appointment, please provide {missing_fields}."


def _reschedule_confirmation_response(state: AgentState) -> str:
    result = state.get("reschedule_result")
    if result is None:
        raise ValueError("Reschedule result is required")
    return (
        f"Your {result['service']} appointment has been rescheduled to "
        f"{result['date']} at {result['time']}."
    )


def _no_active_appointment_response(state: AgentState) -> str:
    intent = state.get("detected_intent")
    action = "reschedule" if intent == "reschedule_appointment" else "cancel"
    return (
        "I don't have an active appointment in this conversation to "
        f"{action} yet."
    )


def _escalation_response(state: AgentState) -> str:
    result = state.get("escalation_result")
    if result is None:
        raise ValueError("Escalation result is required")
    return (
        "I have queued your request for a human specialist. "
        f"Your escalation ID is {result['escalation_id']}."
    )


def _clarification_response(_: AgentState) -> str:
    return (
        "Could you clarify whether you need business information, "
        "appointment help, or a human specialist?"
    )


def _join_fields(fields: list[str]) -> str:
    labels = {
        "service": "service",
        "date": "date",
        "time": "time",
    }
    readable_fields = [labels.get(field, field) for field in fields]
    if not readable_fields:
        raise ValueError("At least one missing field is required")
    if len(readable_fields) == 1:
        return readable_fields[0]
    if len(readable_fields) == 2:
        return " and ".join(readable_fields)
    return ", ".join(readable_fields[:-1]) + f", and {readable_fields[-1]}"


def _business_name(runtime: Runtime[AgentContext]) -> str:
    if runtime.context is None:
        return "this business"

    return runtime.context.business_name or "this business"
