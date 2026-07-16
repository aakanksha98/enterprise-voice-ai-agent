from typing import Literal
from uuid import uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field


class HumanEscalationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str | None = Field(default=None, min_length=1)


class HumanEscalationConfirmation(HumanEscalationRequest):
    escalation_id: str = Field(pattern=r"^ESC-[0-9A-F]{8}$")
    status: Literal["queued"]


@tool("escalate_to_human", args_schema=HumanEscalationRequest)
def mock_human_escalation_tool(
    reason: str | None = None,
) -> dict[str, str | None]:
    """Queue a simulated human handoff without contacting an external service."""
    confirmation = HumanEscalationConfirmation(
        escalation_id=f"ESC-{uuid4().hex[:8].upper()}",
        status="queued",
        reason=reason,
    )
    return confirmation.model_dump()
