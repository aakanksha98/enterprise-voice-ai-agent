from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agent.nodes import (
    mark_ready_for_planning,
    reject_invalid_input,
    validate_input,
)
from backend.app.agent.routing import route_validated_input
from backend.app.agent.state import AgentState


def build_agent_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("validate_input", validate_input)
    graph_builder.add_node("ready_for_planning", mark_ready_for_planning)
    graph_builder.add_node("reject_invalid_input", reject_invalid_input)

    graph_builder.add_edge(START, "validate_input")
    graph_builder.add_conditional_edges(
        "validate_input",
        route_validated_input,
        {
            "ready": "ready_for_planning",
            "invalid": "reject_invalid_input",
        },
    )
    graph_builder.add_edge("ready_for_planning", END)
    graph_builder.add_edge("reject_invalid_input", END)

    return graph_builder.compile(name="enterprise_voice_agent")


agent_graph = build_agent_graph()
