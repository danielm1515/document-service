import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .classifier import OpenAIClassifier
from .config import Settings
from .models import AuditLog, Base

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("document-service")

CLINIC_TZ = ZoneInfo("Asia/Jerusalem")
PATIENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
FROM_ENV = object()  # "build this port from the environment" (the default for classifier/store)


def _israel_today() -> date:
    return datetime.now(CLINIC_TZ).date()


def write_audit(session: Session, *, patient_id: str, operation: str, result: str,
                document_id: str | None, latency_ms: int) -> None:
    """Adds the audit row and commits it with whatever else is pending. The log line carries no
    patient_id (it is in the audit table, which is access-controlled)."""
    logger.info(json.dumps({"operation": operation, "result": result, "document_id": document_id,
                            "latency_ms": latency_ms}))
    session.add(AuditLog(audit_id=str(uuid4()), patient_id=patient_id, operation=operation, result=result,
                         document_id=document_id, latency_ms=latency_ms))
    session.commit()


def create_app(database_url: str | None = None, *, api_key: str | None = None,
               api_auth_enabled: bool | None = None, classifier=FROM_ENV, store=FROM_ENV,
               today: Callable[[], date] | None = None) -> FastAPI:
    settings = Settings()
    url = database_url or settings.database_url
    auth_on = settings.api_auth_enabled if api_auth_enabled is None else api_auth_enabled
    key = settings.api_key if api_key is None else api_key
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Document Service", version="1.0.0", lifespan=lifespan,
                  description="Patients' medical documents: intake, classification and listing.")
    app.state.settings = settings
    app.state.SessionLocal = SessionLocal
    app.state.today = today or _israel_today
    if classifier is FROM_ENV:
        classifier = (OpenAIClassifier(settings.openai_api_key, settings.openai_model)
                      if settings.openai_api_key else None)
    app.state.classifier = classifier
    app.state.store = None if store is FROM_ENV else store  # Task 4 builds it from env

    def authorised(request: Request) -> bool:
        if not auth_on:
            return True
        supplied = request.headers.get("X-API-Key", "")
        return bool(key) and bool(supplied) and secrets.compare_digest(supplied, key)

    app.state.authorised = authorised

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "validation_error"})

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        try:
            with request.app.state.SessionLocal() as session:
                session.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return JSONResponse(status_code=503, content={"status": "degraded", "database": "unavailable"})
        return JSONResponse({
            "status": "ok", "database": "ok",
            "classifier": "configured" if request.app.state.classifier is not None else "not_configured",
            "storage": "configured" if request.app.state.store is not None else "not_configured",
        })

    return app


app = create_app()
