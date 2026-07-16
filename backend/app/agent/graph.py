from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent.booking import execute_booking
from backend.app.agent.cancellation import execute_cancellation
from backend.app.agent.context import AgentContext
from backend.app.agent.escalation import execute_human_escalation
from backend.app.agent.nodes import (
    mark_ready_for_planning,
    plan_request,
    reject_invalid_input,
    retrieve_business_knowledge,
    validate_input,
)
from backend.app.agent.rescheduling import execute_reschedule
from backend.app.agent.response import generate_response
from backend.app.agent.routing import route_planned_intent, route_validated_input
from backend.app.agent.state import AgentState


def build_agent_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState, context_schema=AgentContext)

    graph_builder.add_node("validate_input", validate_input)
    graph_builder.add_node("ready_for_planning", mark_ready_for_planning)
    graph_builder.add_node("reject_invalid_input", reject_invalid_input)
    graph_builder.add_node("planner", plan_request)
    graph_builder.add_node("rag", retrieve_business_knowledge)
    graph_builder.add_node("booking", execute_booking)
    graph_builder.add_node("cancellation", execute_cancellation)
    graph_builder.add_node("reschedule", execute_reschedule)
    graph_builder.add_node("escalation", execute_human_escalation)
    graph_builder.add_node("response", generate_response)

    graph_builder.add_edge(START, "validate_input")
    graph_builder.add_conditional_edges(
        "validate_input",
        route_validated_input,
        {
            "ready": "ready_for_planning",
            "invalid": "reject_invalid_input",
        },
    )
    graph_builder.add_edge("ready_for_planning", "planner")
    graph_builder.add_conditional_edges(
        "planner",
        route_planned_intent,
        {
            "small_talk": "response",
            "rag": "rag",
            "booking": "booking",
            "cancellation": "cancellation",
            "reschedule": "reschedule",
            "escalation": "escalation",
            "deferred": "response",
        },
    )
    graph_builder.add_edge("rag", "response")
    graph_builder.add_edge("booking", "response")
    graph_builder.add_edge("cancellation", "response")
    graph_builder.add_edge("reschedule", "response")
    graph_builder.add_edge("escalation", "response")
    graph_builder.add_edge("reject_invalid_input", "response")
    graph_builder.add_edge("response", END)

    return graph_builder.compile(
        checkpointer=checkpointer,
        name="enterprise_voice_agent",
    )


def build_memory_agent_graph() -> CompiledStateGraph:
    return build_agent_graph(checkpointer=InMemorySaver())


agent_graph = build_agent_graph()
