from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from typing import cast

from langgraph.graph.state import CompiledStateGraph

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import build_memory_agent_graph
from backend.app.agent.planner import create_openai_planner
from backend.app.agent.response import create_openai_response_generator
from backend.app.agent.state import AgentState
from backend.app.business_profiles import (
    BusinessType,
    DEFAULT_BUSINESS_TYPE,
    get_business_profile,
    sanitize_business_name,
)
from backend.app.config import load_local_env
from backend.app.rag.retriever import create_neon_retriever
from backend.app.tools.booking import mock_booking_tool
from backend.app.tools.cancellation import mock_cancellation_tool
from backend.app.tools.escalation import mock_human_escalation_tool
from backend.app.tools.rescheduling import mock_reschedule_tool


@dataclass(frozen=True)
class ConversationService:
    graph: CompiledStateGraph
    context: AgentContext

    def respond(
        self,
        message: str,
        session_id: str,
        *,
        business_type: BusinessType = DEFAULT_BUSINESS_TYPE,
        business_name: str | None = None,
    ) -> AgentState:
        profile = get_business_profile(business_type)
        context = replace(
            self.context,
            business_profile=profile,
            business_name=sanitize_business_name(business_name, profile),
        )
        result = self.graph.invoke(
            {"user_message": message},
            context=context,
            config={"configurable": {"thread_id": f"{business_type}-{session_id}"}},
        )
        state = cast(AgentState, result)
        if not state.get("final_response"):
            raise RuntimeError("Conversation completed without a final response")
        return state


@lru_cache(maxsize=1)
def get_conversation_service() -> ConversationService:
    load_local_env()
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
