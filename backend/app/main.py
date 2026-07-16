from fastapi import FastAPI

from backend.app import __version__
from backend.app.routes.health import router as health_router


API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    application = FastAPI(
        title="Enterprise Voice AI Agent API",
        description="HTTP API for the Enterprise Voice AI Agent.",
        version=__version__,
    )
    application.include_router(health_router, prefix=API_PREFIX)
    return application


app = create_app()
