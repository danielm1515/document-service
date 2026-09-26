import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Path as ApiPath, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException

from .catalog import BY_CODE
from .classifier import ClassifierUnavailable, OpenAIClassifier
from .config import Settings
from .intake import ACCEPTED, DOCUMENT_EXPIRED, DuplicateOf, run_intake
from .magic import CONTENT_TYPE, EXTENSION, sniff_kind
from .models import AuditLog, Base, Document
from .storage import S3ObjectStore, StorageFailed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("document-service")

CLINIC_TZ = ZoneInfo("Asia/Jerusalem")
PATIENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
FROM_ENV = object()  # "build this port from the environment" (the default for classifier/store)
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _israel_today() -> date:
    return datetime.now(CLINIC_TZ).date()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    """SQLite has no native timezone-aware type, so a round-tripped value can come back naive
    even though every write is UTC (`_utc_now`); this always emits an explicit offset."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class AuthBeforeBodyMiddleware:
    """Refuses an unauthenticated or oversized `/api/` request before any body is read.

    FastAPI's routing parses a multipart upload - spooling it to disk, uncapped - before the
    route function runs, so without this an unauthenticated or oversized upload would be fully
    received first. This is a plain ASGI middleware (not `BaseHTTPMiddleware`, which itself reads
    the whole body into memory to build a `Request`), so it inspects only the scope's headers and
    never calls `receive()` on a refusal. The routes keep their own checks too, as defence in
    depth for calls that reach them some other way (e.g. an internal call, or a future change
    here)."""

    def __init__(self, app, *, authorised: Callable[[Request], bool], max_body_bytes: int) -> None:
        self.app = app
        self.authorised = authorised
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        if not self.authorised(request):
            await self._refuse(send, 401, "unauthorized", extra_headers=[(b"www-authenticate", b"ApiKey")])
            return
        if scope["method"] == "POST":
            raw_length = request.headers.get("content-length")
            if raw_length is None:
                await self._refuse(send, 411, "length_required")
                return
            try:
                length = int(raw_length)
            except ValueError:
                await self._refuse(send, 411, "length_required")
                return
            if length > self.max_body_bytes:
                await self._refuse(send, 413, "too_large")
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(send, status: int, error: str, *, extra_headers: list[tuple[bytes, bytes]] = ()) -> None:
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), *extra_headers]})
        await send({"type": "http.response.body", "body": json.dumps({"error": error}).encode()})


def write_audit(session: Session, *, patient_id: str, operation: str, result: str,
                document_id: str | None, latency_ms: int, reason: str | None = None) -> None:
    """Adds the audit row and commits it with whatever else is pending, then logs - only once the
    row is actually durable (minor: log line after commit; a failed commit is never misreported
    as done). The log line carries no patient_id (it is in the audit table, which is
    access-controlled). `reason` is a fixed code, never document content - null unless the
    intake set one (Task 2, decision 2)."""
    session.add(AuditLog(audit_id=str(uuid4()), patient_id=patient_id, operation=operation, result=result,
                         document_id=document_id, latency_ms=latency_ms))
    session.commit()
    logger.info(json.dumps({"operation": operation, "result": result, "document_id": document_id,
                            "reason": reason, "latency_ms": latency_ms}))


def create_app(database_url: str | None = None, *, api_key: str | None = None,
               api_auth_enabled: bool | None = None, classifier=FROM_ENV, store=FROM_ENV,
               today: Callable[[], date] | None = None) -> FastAPI:
    settings = Settings()
    url = database_url or settings.database_url
    auth_on = settings.api_auth_enabled if api_auth_enabled is None else api_auth_enabled
    key = settings.api_key if api_key is None else api_key
    # hide_parameters: a DB error's message (and, from there, any log or traceback that includes
    # it) never carries a bound parameter's value - e.g. a patient_id (I2).
    engine = create_engine(url, hide_parameters=True,
                           connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Document Service", version="1.0.0", lifespan=lifespan,
                  description="Patients' medical documents: intake, classification and listing.")
    app.state.settings = settings
    app.state.engine = engine
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
    app.add_middleware(AuthBeforeBodyMiddleware, authorised=authorised,
                       max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "validation_error"})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Every error is `{"error": code}` (minor): a snake_case rendering of the status phrase -
        404 -> not_found, 405 -> method_not_allowed, and a malformed multipart body, which
        Starlette's own form parser raises as a plain 400, -> bad_request."""
        try:
            code = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_").replace("-", "_")
        except ValueError:
            code = "error"
        headers = exc.headers or None
        return JSONResponse(status_code=exc.status_code, content={"error": code}, headers=headers)

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
            "auth": "configured" if key else "not_configured",
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
            # I2: every bit of this session's work - the duplicate query, the storage-failure
            # audit write and the final commit - is wrapped in one try/except, so any
            # SQLAlchemyError anywhere in it rolls back and answers a plain 503, never a bare 500
            # or a leaked traceback.
            try:
                def accepted_duplicate(sha: str) -> DuplicateOf | None:
                    row = session.execute(select(Document.document_id, Document.document_type,
                                                 Document.document_date).where(
                        Document.patient_id == patient_id, Document.sha256 == sha,
                        Document.result == ACCEPTED).limit(1)).first()
                    return DuplicateOf(row.document_id, row.document_type, row.document_date) if row else None

                try:
                    outcome = run_intake(data, patient_id, classifier=state.classifier, today=state.today(),
                                         is_duplicate=accepted_duplicate, max_bytes=limit)
                except ClassifierUnavailable:
                    # A provider failure is never a verdict on the file (Task 2, decision 1) - the
                    # same pattern as storage_unavailable below: an audit row, no Document row.
                    write_audit(session, patient_id=patient_id, operation="UploadDocument",
                                result="classifier_unavailable", document_id=None, latency_ms=elapsed(started))
                    return JSONResponse(status_code=503, content={"error": "classifier_unavailable"})
                document_id = f"DOC-{secrets.token_hex(6).upper()}"
                object_key = None
                if outcome.result == ACCEPTED:
                    kind = sniff_kind(data) or "pdf"
                    object_key = f"patients/{patient_id}/{document_id}.{EXTENSION[kind]}"
                    try:
                        state.store.put(object_key, data, content_type=CONTENT_TYPE[kind])
                    except StorageFailed:
                        write_audit(session, patient_id=patient_id, operation="UploadDocument",
                                    result="storage_unavailable", document_id=None, latency_ms=elapsed(started))
                        return JSONResponse(status_code=503, content={"error": "storage_unavailable"})
                session.add(Document(document_id=document_id, patient_id=patient_id,
                                     document_type=outcome.document_type, document_date=outcome.document_date,
                                     result=outcome.result, sha256=outcome.sha256, size_bytes=outcome.size_bytes,
                                     s3_object_key=object_key, uploaded_at=_utc_now()))
                write_audit(session, patient_id=patient_id, operation="UploadDocument", result=outcome.result,
                            document_id=document_id, latency_ms=elapsed(started), reason=outcome.reason)
            except SQLAlchemyError:
                session.rollback()
                # If `object_key` was already set, the object is already in S3 and now orphaned:
                # no Document row references it, and GET only lists rows from this table, so the
                # object is never served to anyone.
                return JSONResponse(status_code=503, content={"error": "database_unavailable"})
        content = {
            "document_id": document_id, "document_type": outcome.document_type,
            "document_date": outcome.document_date.isoformat() if outcome.document_date else None,
            "result": outcome.result, "reason": outcome.reason}
        if outcome.duplicate_of:
            content["duplicate_of"] = outcome.duplicate_of
        return JSONResponse(status_code=201, content=content)

    def reported(document: Document, today: date) -> tuple[str, date | None]:
        """I3: validity as of `today`, not as of the upload - the stored row is never touched.
        `valid_until` is null when the row has no catalog type or no document_date; a stored
        ACCEPTED row whose valid_until has passed is reported as DOCUMENT_EXPIRED (inclusive:
        valid_until == today is still ACCEPTED). Every other stored result is reported as-is."""
        doc_type = BY_CODE.get(document.document_type) if document.document_type else None
        valid_until = (document.document_date + timedelta(days=doc_type.max_age_days)
                       if doc_type and document.document_date else None)
        result = document.result
        if result == ACCEPTED and valid_until is not None and valid_until < today:
            result = DOCUMENT_EXPIRED
        return result, valid_until

    @app.get("/api/v1/patients/{patient_id}/documents", tags=["Documents"])
    def list_documents(request: Request, patient_id: str = ApiPath(pattern=PATIENT_ID_PATTERN)) -> JSONResponse:
        """The patient's documents, oldest first (ties broken by insertion order), each with its
        intake result as of today (for CheckDocuments)."""
        if not authorised(request):
            return unauthorised()
        state = request.app.state
        started = time.perf_counter()
        with state.SessionLocal() as session:
            # I2: the query and its audit write are one unit - a failure in either rolls back and
            # answers 503, never a bare 500.
            try:
                documents = list(session.scalars(select(Document).where(Document.patient_id == patient_id)
                                                 .order_by(Document.uploaded_at, Document.seq)))
                write_audit(session, patient_id=patient_id, operation="ListDocuments", result="ok",
                            document_id=None, latency_ms=elapsed(started))
            except SQLAlchemyError:
                session.rollback()
                return JSONResponse(status_code=503, content={"error": "database_unavailable"})
        today = state.today()
        body = []
        for d in documents:
            result, valid_until = reported(d, today)
            body.append({
                "document_id": d.document_id, "document_type": d.document_type,
                "document_date": d.document_date.isoformat() if d.document_date else None,
                "result": result, "valid_until": valid_until.isoformat() if valid_until else None,
                "uploaded_at": _iso_utc(d.uploaded_at)})
        return JSONResponse({"documents": body})

    return app


app = create_app()
