from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field


class BookingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    service: str = Field(min_length=1)
    date: str = Field(min_length=1)
    time: str = Field(min_length=1)


class BookingConfirmation(BookingRequest):
    status: Literal["confirmed"]


@tool("book_appointment", args_schema=BookingRequest)
def mock_booking_tool(
    service: str,
    date: str,
    time: str,
) -> dict[str, str]:
    """Create a simulated appointment without calling an external service."""
    confirmation = BookingConfirmation(
        status="confirmed",
        service=service,
        date=date,
        time=time,
    )
    return confirmation.model_dump()
