from dataclasses import dataclass

from backend.app.agent.planner import PlannerRunnable
from backend.app.rag.retriever import RAGRetriever


@dataclass(frozen=True)
class AgentContext:
    planner: PlannerRunnable
    rag_retriever: RAGRetriever | None = None
