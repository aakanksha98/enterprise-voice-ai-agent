from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field


class RescheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    appointment_id: str = Field(min_length=1)
    new_date: str = Field(min_length=1)
    new_time: str = Field(min_length=1)


class RescheduleConfirmation(RescheduleRequest):
    status: Literal["rescheduled"]


@tool("reschedule_appointment", args_schema=RescheduleRequest)
def mock_reschedule_tool(
    appointment_id: str,
    new_date: str,
    new_time: str,
) -> dict[str, str]:
    """Simulate rescheduling without calling an external appointment service."""
    confirmation = RescheduleConfirmation(
        appointment_id=appointment_id,
        status="rescheduled",
        new_date=new_date,
        new_time=new_time,
    )
    return confirmation.model_dump()
