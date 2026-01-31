"""Uvicorn server entry point."""

import uvicorn

from app.core.config import settings


def main() -> None:
    """
    Run the FastAPI application with Uvicorn.

    This is the main entry point for running the server.
    """
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
