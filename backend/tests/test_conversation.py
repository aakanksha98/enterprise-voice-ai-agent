from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import build_memory_agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.main import app
from backend.app.services.conversation import (
    ConversationService,
    get_conversation_service,
)
from backend.app.tools.booking import BookingRequest


CLARIFICATION_RESPONSE = (
    "Could you clarify whether you need business information, "
    "appointment help, or a human specialist?"
)


@pytest.fixture
def planner_inputs() -> list[dict[str, str]]:
    return []


@pytest.fixture
def conversation_client(
    planner_inputs: list[dict[str, str]],
) -> Iterator[TestClient]:
    decision = PlannerDecision(
        intent="clarification",
        confidence=0.8,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )

    def plan(planner_input: dict[str, str]) -> PlannerDecision:
        planner_inputs.append(planner_input)
        return decision

    service = ConversationService(
        graph=build_memory_agent_graph(),
        context=AgentContext(planner=RunnableLambda(plan)),
    )
    app.dependency_overrides[get_conversation_service] = lambda: service
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_conversation_endpoint_returns_final_response(
    conversation_client: TestClient,
    planner_inputs: list[dict[str, str]],
) -> None:
    response = conversation_client.post(
        "/api/v1/conversation",
        json={
            "message": "  Please help me  ",
            "session_id": "session-123",
            "business_type": "dental",
            "business_name": "Happy Tooth Studio",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": CLARIFICATION_RESPONSE,
        "business_type": "dental",
        "business_name": "Happy Tooth Studio",
        "intent": "clarification",
        "workflow_stage": "planned",
        "slots": {},
        "missing_fields": [],
        "active_appointment": None,
    }
    assert planner_inputs == [
        {
            "user_message": "Please help me",
            "conversation_history": "No prior conversation.",
            "business_type": "Dental Clinic",
            "supported_services": (
                "dental cleaning, dental exam, teeth whitening, filling, "
                "emergency dental visit"
            ),
        }
    ]


def test_conversation_endpoint_reuses_session_memory(
    conversation_client: TestClient,
    planner_inputs: list[dict[str, str]],
) -> None:
    for message in ("First request", "Follow-up request"):
        response = conversation_client.post(
            "/api/v1/conversation",
            json={
                "message": message,
                "session_id": "memory-session",
                "business_type": "dental",
                "business_name": "Any Dental Name",
            },
        )
        assert response.status_code == 200

    assert planner_inputs == [
        {
            "user_message": "First request",
            "conversation_history": "No prior conversation.",
            "business_type": "Dental Clinic",
            "supported_services": (
                "dental cleaning, dental exam, teeth whitening, filling, "
                "emergency dental visit"
            ),
        },
        {
            "user_message": "Follow-up request",
            "conversation_history": (
                "user: First request\n"
                f"assistant: {CLARIFICATION_RESPONSE}"
            ),
            "business_type": "Dental Clinic",
            "supported_services": (
                "dental cleaning, dental exam, teeth whitening, filling, "
                "emergency dental visit"
            ),
        },
    ]


def test_conversation_endpoint_returns_tool_reference() -> None:
    decision = PlannerDecision(
        intent="book_appointment",
        confidence=0.98,
        small_talk_topic=None,
        slots=PlannerSlots(
            service="haircut",
            date="Friday",
            time="4 PM",
            escalation_reason=None,
        ),
    )

    @tool("api_booking", args_schema=BookingRequest)
    def api_booking(service: str, date: str, time: str) -> dict[str, str]:
        """Return a deterministic booking for the conversation API test."""
        return {
            "status": "confirmed",
            "service": service,
            "date": date,
            "time": time,
        }

    service = ConversationService(
        graph=build_memory_agent_graph(),
        context=AgentContext(
            planner=RunnableLambda(lambda _: decision),
            booking_tool=api_booking,
        ),
    )
    app.dependency_overrides[get_conversation_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/api/v1/conversation",
            json={
                "message": "Book a haircut Friday at 4 PM",
                "session_id": "booking-session",
                "business_type": "salon",
                "business_name": "Luxe Hair Studio",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["business_type"] == "salon"
    assert response.json()["business_name"] == "Luxe Hair Studio"
    assert response.json()["active_appointment"] == {
        "service": "haircut",
        "date": "Friday",
        "time": "4 PM",
        "status": "confirmed",
    }
    assert response.json()["slots"] == {
        "service": "haircut",
        "date": "Friday",
        "time": "4 PM",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "message": "   ",
            "session_id": "session-123",
            "business_type": "dental",
            "business_name": "Happy Tooth Studio",
        },
        {
            "message": "Hello",
            "session_id": "bad id",
            "business_type": "dental",
            "business_name": "Happy Tooth Studio",
        },
        {
            "message": "Hello",
            "session_id": "short",
            "business_type": "dental",
            "business_name": "Happy Tooth Studio",
        },
        {
            "message": "Hello",
            "session_id": "session-123",
            "business_name": "Happy Tooth Studio",
        },
        {
            "message": "Hello",
            "session_id": "session-123",
            "business_type": "dental",
        },
        {
            "message": "Hello",
            "session_id": "session-123",
            "business_type": "dental",
            "business_name": "   ",
        },
    ],
)
def test_conversation_endpoint_validates_request(
    payload: dict[str, str],
    conversation_client: TestClient,
) -> None:
    response = conversation_client.post("/api/v1/conversation", json=payload)

    assert response.status_code == 422


def test_frontend_is_served_by_fastapi() -> None:
    client = TestClient(app)

    page_response = client.get("/")
    script_response = client.get("/app.js")

    assert page_response.status_code == 200
    assert "Enterprise Voice AI Agent" in page_response.text
    assert script_response.status_code == 200
    assert "/api/v1/conversation" in script_response.text
