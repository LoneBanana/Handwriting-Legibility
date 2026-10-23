"""Application factory and the ``legible-api`` entry point.

Development: ``legible-api --reload`` (API on :8000) and ``npm run dev`` in ``web/``.
Production: build the web app (``npm run build``) and the API serves it from ``web/dist``
on the same origin, e.g. ``uvicorn --factory legible.api.main:create_app``.
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .db import make_engine, make_sessionmaker
from .migrate import upgrade
from .routers import auth, converse, handwriting, literacy, numeracy, progress
from .services.handwriting import HandwritingService
from .services.llm import build_llm
from .services.personalization import PersonalizationWorker
from .settings import Settings, get_settings

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.resolved_database_url)
    if settings.auto_migrate:
        upgrade(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.personalizer.shutdown()
        await app.state.llm.aclose()
        engine.dispose()

    app = FastAPI(
        title="Legible API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessionmaker = make_sessionmaker(engine)
    app.state.secret_key = settings.secret_key()
    app.state.llm = build_llm(settings)
    app.state.handwriting = HandwritingService(settings)
    app.state.personalizer = PersonalizationWorker(settings, app.state.handwriting, app.state.sessionmaker)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Timezone"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes the offending input back, which crashes on
        # values JSON cannot encode (NaN, Infinity) and can repeat sensitive input.
        errors = [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    for module in (auth, progress, handwriting, literacy, numeracy, converse):
        app.include_router(module.router)

    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    dist = settings.frontend_dist.resolve()
    index = dist / "index.html"
    if not settings.serve_frontend or not index.is_file():
        return
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def single_page_app(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, detail="Not found")
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def run() -> None:
    parser = argparse.ArgumentParser(description="Run the Legible API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    uvicorn.run(
        "legible.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).resolve().parents[1])] if args.reload else None,
        workers=None if args.reload else args.workers,
    )
