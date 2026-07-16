from dataclasses import dataclass

from langchain_core.tools import BaseTool

from backend.app.agent.planner import PlannerRunnable
from backend.app.agent.response import ResponseRunnable
from backend.app.rag.retriever import RAGRetriever


@dataclass(frozen=True)
class AgentContext:
    planner: PlannerRunnable
    rag_retriever: RAGRetriever | None = None
    booking_tool: BaseTool | None = None
    cancellation_tool: BaseTool | None = None
    reschedule_tool: BaseTool | None = None
    escalation_tool: BaseTool | None = None
    response_generator: ResponseRunnable | None = None
