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

from fastapi import FastAPI, File, Path as ApiPath, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .classifier import OpenAIClassifier
from .config import Settings
from .intake import ACCEPTED, run_intake
from .models import AuditLog, Base, Document
from .storage import S3ObjectStore, StorageFailed

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
    if store is FROM_ENV:
        store = S3ObjectStore(settings.s3_bucket, settings.aws_region) if settings.s3_bucket else None
    app.state.store = store

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

    def unauthorised() -> JSONResponse:
        return JSONResponse(status_code=401, content={"error": "unauthorized"}, headers={"WWW-Authenticate": "ApiKey"})

    def elapsed(started: float) -> int:
        return round((time.perf_counter() - started) * 1000)

    @app.post("/api/v1/patients/{patient_id}/documents", status_code=201, tags=["Documents"])
    def upload_document(request: Request, patient_id: str = ApiPath(pattern=PATIENT_ID_PATTERN),
                        file: UploadFile = File(...)) -> JSONResponse:
        """Checks and classifies one PDF; stores it only if accepted (design §4.2)."""
        if not authorised(request):
            return unauthorised()
        state = request.app.state
        if state.classifier is None or state.store is None:
            return JSONResponse(status_code=503, content={"error": "service_not_configured"})
        started = time.perf_counter()
        limit = state.settings.max_upload_bytes
        data = file.file.read(limit + 1)  # one byte over the limit is enough to know it is too big
        with state.SessionLocal() as session:
            def accepted_duplicate(sha: str) -> bool:
                return session.scalar(select(Document.document_id).where(
                    Document.patient_id == patient_id, Document.sha256 == sha, Document.result == ACCEPTED)
                    .limit(1)) is not None

            outcome = run_intake(data, patient_id, classifier=state.classifier, today=state.today(),
                                 is_duplicate=accepted_duplicate, max_bytes=limit)
            document_id = f"DOC-{secrets.token_hex(6).upper()}"
            key = None
            if outcome.result == ACCEPTED:
                key = f"patients/{patient_id}/{document_id}.pdf"
                try:
                    state.store.put(key, data)
                except StorageFailed:
                    write_audit(session, patient_id=patient_id, operation="UploadDocument",
                                result="storage_unavailable", document_id=None, latency_ms=elapsed(started))
                    return JSONResponse(status_code=503, content={"error": "storage_unavailable"})
            session.add(Document(document_id=document_id, patient_id=patient_id, document_type=outcome.document_type,
                                 document_date=outcome.document_date, result=outcome.result, sha256=outcome.sha256,
                                 size_bytes=outcome.size_bytes, s3_object_key=key))
            write_audit(session, patient_id=patient_id, operation="UploadDocument", result=outcome.result,
                        document_id=document_id, latency_ms=elapsed(started))
        return JSONResponse(status_code=201, content={
            "document_id": document_id, "document_type": outcome.document_type,
            "document_date": outcome.document_date.isoformat() if outcome.document_date else None,
            "result": outcome.result})

    @app.get("/api/v1/patients/{patient_id}/documents", tags=["Documents"])
    def list_documents(request: Request, patient_id: str = ApiPath(pattern=PATIENT_ID_PATTERN)) -> JSONResponse:
        """The patient's documents, oldest first, with their intake results (for CheckDocuments)."""
        if not authorised(request):
            return unauthorised()
        started = time.perf_counter()
        with request.app.state.SessionLocal() as session:
            documents = list(session.scalars(select(Document).where(Document.patient_id == patient_id)
                                             .order_by(Document.uploaded_at, Document.document_id)))
            write_audit(session, patient_id=patient_id, operation="ListDocuments", result="ok",
                        document_id=None, latency_ms=elapsed(started))
        return JSONResponse({"documents": [{
            "document_id": d.document_id, "document_type": d.document_type,
            "document_date": d.document_date.isoformat() if d.document_date else None,
            "result": d.result, "uploaded_at": d.uploaded_at.isoformat()} for d in documents]})

    return app


app = create_app()
