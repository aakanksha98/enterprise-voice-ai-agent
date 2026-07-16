import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import build_memory_agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.tools.booking import BookingRequest


def booking_decision(
    *,
    service: str | None = None,
    date: str | None = None,
    time: str | None = None,
) -> PlannerDecision:
    return PlannerDecision(
        intent="book_appointment",
        confidence=0.95,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=service,
            date=date,
            time=time,
            appointment_id=None,
            escalation_reason=None,
        ),
    )


def test_memory_carries_slots_across_an_incomplete_booking() -> None:
    planner_inputs: list[dict[str, str]] = []
    tool_calls: list[dict[str, str]] = []

    def plan(planner_input: dict[str, str]) -> PlannerDecision:
        planner_inputs.append(planner_input)
        if planner_input["user_message"] == "Book a haircut tomorrow":
            return booking_decision(service="haircut", date="tomorrow")
        return booking_decision(time="2 PM")

    @tool("record_memory_booking", args_schema=BookingRequest)
    def record_memory_booking(
        service: str,
        date: str,
        time: str,
    ) -> dict[str, str]:
        """Record a booking assembled across conversation turns."""
        tool_calls.append({"service": service, "date": date, "time": time})
        return {
            "appointment_id": "APT-1234ABCD",
            "status": "confirmed",
            "service": service,
            "date": date,
            "time": time,
        }

    graph = build_memory_agent_graph()
    context = AgentContext(
        planner=RunnableLambda(plan),
        booking_tool=record_memory_booking,
    )
    config = {"configurable": {"thread_id": "booking-thread"}}

    first_result = graph.invoke(
        {"user_message": "Book a haircut tomorrow"},
        context=context,
        config=config,
    )
    second_result = graph.invoke(
        {"user_message": "At 2 PM"},
        context=context,
        config=config,
    )

    assert first_result["workflow_stage"] == "booking_information_required"
    assert first_result["missing_booking_slots"] == ["time"]
    assert planner_inputs == [
        {
            "user_message": "Book a haircut tomorrow",
            "conversation_history": "No prior conversation.",
        },
        {
            "user_message": "At 2 PM",
            "conversation_history": (
                "user: Book a haircut tomorrow\n"
                "assistant: To book the appointment, please provide time."
            ),
        },
    ]
    assert tool_calls == [
        {"service": "haircut", "date": "tomorrow", "time": "2 PM"}
    ]
    assert second_result["workflow_stage"] == "appointment_booked"
    assert second_result["conversation_history"] == [
        {"role": "user", "content": "Book a haircut tomorrow"},
        {
            "role": "assistant",
            "content": "To book the appointment, please provide time.",
        },
        {"role": "user", "content": "At 2 PM"},
        {
            "role": "assistant",
            "content": (
                "Your haircut appointment is booked for tomorrow at 2 PM. "
                "Your appointment ID is APT-1234ABCD."
            ),
        },
    ]


def test_memory_isolates_conversation_threads() -> None:
    graph = build_memory_agent_graph()
    decision = PlannerDecision(
        intent="clarification",
        confidence=0.8,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            appointment_id=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(planner=RunnableLambda(lambda _: decision))

    graph.invoke(
        {"user_message": "First thread message"},
        context=context,
        config={"configurable": {"thread_id": "thread-a"}},
    )
    second_thread = graph.invoke(
        {"user_message": "Second thread message"},
        context=context,
        config={"configurable": {"thread_id": "thread-b"}},
    )

    assert second_thread["conversation_history"] == [
        {"role": "user", "content": "Second thread message"},
        {
            "role": "assistant",
            "content": (
                "Could you clarify whether you need business information, "
                "appointment help, or a human specialist?"
            ),
        },
    ]


def test_memory_graph_requires_thread_identifier() -> None:
    graph = build_memory_agent_graph()

    with pytest.raises(ValueError, match="thread_id"):
        graph.invoke({"user_message": "Hello"})


def test_new_turn_clears_stale_terminal_outputs() -> None:
    decisions = iter(
        [
            PlannerDecision(
                intent="small_talk",
                confidence=0.98,
                small_talk_topic="greeting",
                slots=PlannerSlots(
                    service=None,
                    date=None,
                    time=None,
                    appointment_id=None,
                    escalation_reason=None,
                ),
            ),
            PlannerDecision(
                intent="clarification",
                confidence=0.7,
                small_talk_topic=None,
                slots=PlannerSlots(
                    service=None,
                    date=None,
                    time=None,
                    appointment_id=None,
                    escalation_reason=None,
                ),
            ),
        ]
    )
    graph = build_memory_agent_graph()
    context = AgentContext(planner=RunnableLambda(lambda _: next(decisions)))
    config = {"configurable": {"thread_id": "reset-thread"}}

    first_result = graph.invoke(
        {"user_message": "Hello"},
        context=context,
        config=config,
    )
    second_result = graph.invoke(
        {"user_message": "Something unclear"},
        context=context,
        config=config,
    )

    assert first_result["small_talk_topic"] == "greeting"
    assert second_result["workflow_stage"] == "planned"
    assert second_result["small_talk_topic"] is None
    assert second_result["retrieved_documents"] == []
    assert second_result["escalation_result"] is None
