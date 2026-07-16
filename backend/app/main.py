from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app import __version__
from backend.app.routes.conversation import router as conversation_router
from backend.app.routes.health import router as health_router


API_PREFIX = "/api/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIRECTORY = (
    PROJECT_ROOT / "public"
    if (PROJECT_ROOT / "public").exists()
    else PROJECT_ROOT / "frontend"
)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Enterprise Voice AI Agent API",
        description="HTTP API for the Enterprise Voice AI Agent.",
        version=__version__,
    )
    application.include_router(health_router, prefix=API_PREFIX)
    application.include_router(conversation_router, prefix=API_PREFIX)
    application.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIRECTORY, html=True),
        name="frontend",
    )
    return application


app = create_app()
