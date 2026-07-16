from dataclasses import dataclass
from functools import lru_cache
from typing import cast

from langgraph.graph.state import CompiledStateGraph

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import build_memory_agent_graph
from backend.app.agent.planner import create_openai_planner
from backend.app.agent.response import create_openai_response_generator
from backend.app.agent.state import AgentState
from backend.app.rag.retriever import create_neon_retriever
from backend.app.tools.booking import mock_booking_tool
from backend.app.tools.cancellation import mock_cancellation_tool
from backend.app.tools.escalation import mock_human_escalation_tool
from backend.app.tools.rescheduling import mock_reschedule_tool


@dataclass(frozen=True)
class ConversationService:
    graph: CompiledStateGraph
    context: AgentContext

    def respond(self, message: str, session_id: str) -> AgentState:
        result = self.graph.invoke(
            {"user_message": message},
            context=self.context,
            config={"configurable": {"thread_id": session_id}},
        )
        state = cast(AgentState, result)
        if not state.get("final_response"):
            raise RuntimeError("Conversation completed without a final response")
        return state


@lru_cache(maxsize=1)
def get_conversation_service() -> ConversationService:
    context = AgentContext(
        planner=create_openai_planner(),
        rag_retriever=create_neon_retriever(),
        booking_tool=mock_booking_tool,
        cancellation_tool=mock_cancellation_tool,
        reschedule_tool=mock_reschedule_tool,
        escalation_tool=mock_human_escalation_tool,
        response_generator=create_openai_response_generator(),
    )
    return ConversationService(
        graph=build_memory_agent_graph(),
        context=context,
    )
