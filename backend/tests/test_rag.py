import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent


def rag_decision() -> PlannerDecision:
    return PlannerDecision(
        intent="rag",
        confidence=0.93,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )


def test_rag_intent_retrieves_and_serializes_business_knowledge() -> None:
    retrieval_queries: list[dict[str, str]] = []
    response_inputs: list[dict[str, str]] = []

    def retrieve(request: dict[str, str]) -> list[Document]:
        retrieval_queries.append(request)
        return [
            Document(
                page_content="  Haircuts start at $35.  ",
                metadata={
                    "business_type": "dental",
                    "source": "service-catalog",
                    "category": "pricing",
                    "similarity": 0.91,
                },
            ),
            Document(page_content="Open Monday through Friday."),
            Document(page_content="   ", metadata={"source": "ignored"}),
        ]

    context = AgentContext(
        planner=RunnableLambda(lambda _: rag_decision()),
        rag_retriever=RunnableLambda(retrieve),
        response_generator=RunnableLambda(
            lambda response_input: (
                response_inputs.append(response_input)
                or "Haircuts start at $35, and we are open Monday through Friday."
            )
        ),
    )
    result = agent_graph.invoke(
        {"user_message": "  What does a haircut cost?  "},
        context=context,
    )

    assert retrieval_queries == [
        {"query": "What does a haircut cost?", "business_type": "dental"}
    ]
    assert result == {
        "user_message": "  What does a haircut cost?  ",
        "conversation_history": [
            {"role": "user", "content": "What does a haircut cost?"},
            {
                "role": "assistant",
                "content": (
                    "Haircuts start at $35, and we are open Monday through Friday."
                ),
            },
        ],
        "normalized_message": "What does a haircut cost?",
        "input_status": "valid",
        "workflow_stage": "knowledge_retrieved",
        "detected_intent": "rag",
        "planner_confidence": 0.93,
        "extracted_slots": {},
        "small_talk_topic": None,
        "retrieval_query": "What does a haircut cost?",
        "retrieved_documents": [
            {
                "content": "Haircuts start at $35.",
                "source": "service-catalog",
                "category": "pricing",
                "business_type": "dental",
                "similarity": 0.91,
            },
            {
                "content": "Open Monday through Friday.",
                "source": "business_knowledge",
            },
        ],
        "final_response": (
            "Haircuts start at $35, and we are open Monday through Friday."
        ),
    }
    assert response_inputs == [
        {
            "user_message": "What does a haircut cost?",
            "retrieved_context": (
                "Source: service-catalog\nHaircuts start at $35.\n\n"
                "Source: business_knowledge\nOpen Monday through Friday."
            ),
        }
    ]


def test_rag_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: rag_decision()),
        rag_retriever=RunnableLambda(lambda _: []),
    )
    updates = agent_graph.stream(
        {"user_message": "What services are available?"},
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "rag",
        "response",
    ]


def test_rag_intent_requires_retriever_context() -> None:
    context = AgentContext(planner=RunnableLambda(lambda _: rag_decision()))

    with pytest.raises(RuntimeError, match="RAG retriever context is required"):
        agent_graph.invoke(
            {"user_message": "What is your refund policy?"},
            context=context,
        )


def test_planned_intent_router_selects_rag_route() -> None:
    assert route_planned_intent(
        {"user_message": "What are your prices?", "detected_intent": "rag"}
    ) == "rag"
