from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field


class CancellationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    service: str = Field(min_length=1)
    date: str = Field(min_length=1)
    time: str = Field(min_length=1)


class CancellationConfirmation(CancellationRequest):
    status: Literal["cancelled"]


@tool("cancel_appointment", args_schema=CancellationRequest)
def mock_cancellation_tool(
    service: str,
    date: str,
    time: str,
) -> dict[str, str]:
    """Simulate cancellation without calling an external appointment service."""
    confirmation = CancellationConfirmation(
        service=service,
        date=date,
        time=time,
        status="cancelled",
    )
    return confirmation.model_dump()
