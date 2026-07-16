from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app import __version__


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Check API health")
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="enterprise-voice-ai-agent",
        version=__version__,
    )
