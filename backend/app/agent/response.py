from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from backend.app.agent.state import AgentState, AgentStateUpdate


if TYPE_CHECKING:
    from backend.app.agent.context import AgentContext


DEFAULT_RESPONSE_MODEL = "gpt-5.4-mini"
ResponseRunnable = Runnable[dict[str, str], str]


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
        "faq_answered": _faq_response,
        "booking_information_required": _booking_information_response,
        "appointment_booked": _booking_confirmation_response,
        "cancellation_information_required": _cancellation_information_response,
        "appointment_cancelled": _cancellation_confirmation_response,
        "reschedule_information_required": _reschedule_information_response,
        "appointment_rescheduled": _reschedule_confirmation_response,
        "human_escalation_queued": _escalation_response,
        "planned": _clarification_response,
    }

    if workflow_stage == "knowledge_retrieved":
        return _rag_response(state, runtime)

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


def _faq_response(state: AgentState) -> str:
    draft_response = state.get("draft_response")
    if not draft_response:
        raise ValueError("FAQ response text is required")
    return draft_response


def _booking_information_response(state: AgentState) -> str:
    missing_fields = _join_fields(state.get("missing_booking_slots", []))
    return f"To book the appointment, please provide {missing_fields}."


def _booking_confirmation_response(state: AgentState) -> str:
    result = state.get("booking_result")
    if result is None:
        raise ValueError("Booking result is required")
    return (
        f"Your {result['service']} appointment is booked for {result['date']} "
        f"at {result['time']}. Your appointment ID is {result['appointment_id']}."
    )


def _cancellation_information_response(_: AgentState) -> str:
    return "Please provide your appointment ID so I can cancel the appointment."


def _cancellation_confirmation_response(state: AgentState) -> str:
    result = state.get("cancellation_result")
    if result is None:
        raise ValueError("Cancellation result is required")
    return f"Appointment {result['appointment_id']} has been cancelled."


def _reschedule_information_response(state: AgentState) -> str:
    missing_fields = _join_fields(state.get("missing_reschedule_slots", []))
    return f"To reschedule the appointment, please provide {missing_fields}."


def _reschedule_confirmation_response(state: AgentState) -> str:
    result = state.get("reschedule_result")
    if result is None:
        raise ValueError("Reschedule result is required")
    return (
        f"Appointment {result['appointment_id']} has been rescheduled to "
        f"{result['new_date']} at {result['new_time']}."
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
        "appointment_id": "appointment ID",
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
