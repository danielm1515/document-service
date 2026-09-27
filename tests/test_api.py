import asyncio
import io
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.classifier import ClassifierUnavailable, FakeClassifier
from app.main import MULTIPART_OVERHEAD_BYTES, create_app
from app.models import AuditLog, Document
from app.storage import InMemoryObjectStore, StorageFailed

FIXTURES = Path(__file__).parent / "fixtures"
KEY = "test-key"
H = {"X-API-Key": KEY}
MAX_BODY_BYTES = 10 * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES  # app.config.Settings.max_upload_bytes + overhead


def pdf(year, name):
    return (FIXTURES / year / f"{name}.pdf").read_bytes()


def make(tmp_path, *, store=None, classifier=None, **kwargs):
    store = InMemoryObjectStore() if store is None else store
    app = create_app(f"sqlite:///{(tmp_path / 'd.db').as_posix()}", api_key=KEY,
                     classifier=classifier or FakeClassifier(), store=store, today=lambda: date(2026, 9, 22), **kwargs)
    return app, store


def upload(client, name, year="2026", patient="P-10041", headers=H):
    return client.post(f"/api/v1/patients/{patient}/documents", headers=headers,
                       files={"file": (f"{name}.pdf", pdf(year, name), "application/pdf")})


def rows(app, model):
    with app.state.SessionLocal() as session:
        return list(session.scalars(select(model)))


def test_an_accepted_document_is_stored_and_listed(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        response = upload(client, "cbc")
        listing = client.get("/api/v1/patients/P-10041/documents", headers=H).json()
    assert response.status_code == 201
    body = response.json()
    assert re.fullmatch(r"DOC-[0-9A-F]{12}", body["document_id"])
    assert body == {"document_id": body["document_id"], "document_type": "CBC",
                    "document_date": "2026-09-15", "result": "ACCEPTED", "reason": None,
                    "llm_usage": FakeClassifier.TEXT_USAGE.as_json()}
    assert list(store.objects) == [f"patients/P-10041/{body['document_id']}.pdf"]
    assert store.objects[f"patients/P-10041/{body['document_id']}.pdf"] == pdf("2026", "cbc")
    [doc] = listing["documents"]
    assert {k: doc[k] for k in ("document_id", "document_type", "document_date", "result")} == {
        "document_id": body["document_id"], "document_type": "CBC", "document_date": "2026-09-15", "result": "ACCEPTED"}
    assert doc["valid_until"] == "2026-12-14"  # 2026-09-15 + CBC's 90 days
    assert "uploaded_at" in doc
    # set in Python (not a DB default), stored with microseconds, emitted with a UTC offset.
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{2}:\d{2}$", doc["uploaded_at"])


@pytest.mark.parametrize("name, year, result", [
    ("electricity_bill", "2026", "NON_MEDICAL_DOCUMENT"),
    ("cbc", "2024", "DOCUMENT_EXPIRED"),
])
def test_a_rejected_document_is_recorded_but_never_stored(tmp_path, name, year, result):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        body = upload(client, name, year).json()
    assert body["result"] == result
    assert store.objects == {}
    [row] = rows(app, Document)
    assert row.result == result and row.s3_object_key is None


def test_the_same_accepted_file_again_is_a_duplicate_and_a_rejected_one_is_checked_afresh(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        assert upload(client, "cbc").json()["result"] == "ACCEPTED"
        assert upload(client, "cbc").json()["result"] == "DUPLICATE_DOCUMENT"
        assert upload(client, "electricity_bill").json()["result"] == "NON_MEDICAL_DOCUMENT"
        assert upload(client, "electricity_bill").json()["result"] == "NON_MEDICAL_DOCUMENT"
        # Another patient uploading the same file is not a duplicate of the first patient's.
        assert upload(client, "cbc", patient="P-20000").json()["result"] == "ACCEPTED"
    assert len(store.objects) == 2


def test_a_duplicate_upload_reports_the_accepted_original(tmp_path):
    # I4: the retry can be treated as the document already delivered.
    app, store = make(tmp_path)
    with TestClient(app) as client:
        first = upload(client, "cbc").json()
        assert "duplicate_of" not in first
        second = upload(client, "cbc").json()
    assert second["result"] == "DUPLICATE_DOCUMENT"
    assert second["duplicate_of"] == first["document_id"]
    assert second["document_type"] == "CBC" and second["document_date"] == "2026-09-15"
    assert len(store.objects) == 1


# --- I3: the listing reports validity as of app.state.today(), the stored row is never touched. ---

def test_listing_reports_expiry_as_of_today_without_touching_the_stored_row(tmp_path):
    box = {"value": date(2026, 9, 22)}
    app = create_app(f"sqlite:///{(tmp_path / 'd.db').as_posix()}", api_key=KEY,
                     classifier=FakeClassifier(), store=InMemoryObjectStore(), today=lambda: box["value"])
    with TestClient(app) as client:
        upload(client, "cbc")

        def listed():
            return client.get("/api/v1/patients/P-10041/documents", headers=H).json()["documents"][0]

        assert listed()["result"] == "ACCEPTED" and listed()["valid_until"] == "2026-12-14"

        box["value"] = date(2026, 12, 14)  # inclusive: valid_until == today is still ACCEPTED
        assert listed()["result"] == "ACCEPTED"

        box["value"] = date(2026, 12, 15)  # one day past valid_until
        expired = listed()
        assert expired["result"] == "DOCUMENT_EXPIRED" and expired["valid_until"] == "2026-12-14"

    [row] = rows(app, Document)
    assert row.result == "ACCEPTED"  # the stored row itself is never modified


def test_valid_until_is_null_without_a_type_or_a_date(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        upload(client, "electricity_bill")
        [doc] = client.get("/api/v1/patients/P-10041/documents", headers=H).json()["documents"]
    assert doc["document_type"] is None and doc["valid_until"] is None


def test_the_six_demo_files_give_the_designs_results(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        results = {name: upload(client, name).json() for name in
                   ("cbc", "coagulation", "ecg", "urinalysis", "preop_summary", "electricity_bill")}
    assert {n: (r["result"], r["document_type"]) for n, r in results.items()} == {
        "cbc": ("ACCEPTED", "CBC"), "coagulation": ("ACCEPTED", "COAGULATION_TESTS"), "ecg": ("ACCEPTED", "ECG"),
        "urinalysis": ("ACCEPTED", "URINALYSIS"), "preop_summary": ("ACCEPTED", "PREOP_SUMMARY"),
        "electricity_bill": ("NON_MEDICAL_DOCUMENT", None)}


def test_a_patient_sees_only_their_own_documents(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        upload(client, "cbc", patient="P-10041")
        upload(client, "ecg", patient="P-20000")
        mine = client.get("/api/v1/patients/P-10041/documents", headers=H).json()["documents"]
    assert [d["document_type"] for d in mine] == ["CBC"]


def test_every_call_needs_the_key(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        assert upload(client, "cbc", headers={}).status_code == 401
        assert upload(client, "cbc", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/v1/patients/P-10041/documents").status_code == 401
    assert store.objects == {} and rows(app, Document) == []


def test_an_invalid_patient_id_is_refused(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        assert upload(client, "cbc", patient="P 1").status_code == 400
        assert client.get("/api/v1/patients/%20/documents", headers=H).status_code == 400


def test_a_missing_file_is_refused(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/v1/patients/P-10041/documents", headers=H).status_code == 400


def test_without_a_classifier_or_a_store_nothing_is_accepted(tmp_path):
    for kwargs in ({"classifier": None}, {"store": None}):
        app = create_app(f"sqlite:///{(tmp_path / f'{len(kwargs)}{list(kwargs)[0]}.db').as_posix()}", api_key=KEY,
                         classifier=kwargs.get("classifier", FakeClassifier()), store=kwargs.get("store", InMemoryObjectStore()),
                         today=lambda: date(2026, 9, 22))
        with TestClient(app) as client:
            response = upload(client, "cbc")
        assert response.status_code == 503 and response.json() == {"error": "service_not_configured"}
        assert rows(app, Document) == []


def test_a_storage_failure_stores_nothing_and_says_so(tmp_path):
    class Failing:
        def put(self, key, data, content_type="application/pdf"):
            raise StorageFailed("AccessDenied")
    app, _ = make(tmp_path, store=Failing())
    with TestClient(app) as client:
        response = upload(client, "cbc")
    assert response.status_code == 503 and response.json() == {"error": "storage_unavailable"}
    assert rows(app, Document) == []
    assert [(a.operation, a.result) for a in rows(app, AuditLog)] == [("UploadDocument", "storage_unavailable")]


# --- Task 2, decision 1: a classifier provider failure is 503 classifier_unavailable, the same
# pattern as storage_unavailable above - never a verdict on the file, no Document row. ---

class _UnavailableClassifier:
    def classify(self, text):
        raise ClassifierUnavailable("api:RateLimitError")

    def classify_images(self, images):
        raise ClassifierUnavailable("api:RateLimitError")


def test_a_classifier_provider_failure_is_503_with_no_document_row(tmp_path):
    app, store = make(tmp_path, classifier=_UnavailableClassifier())
    with TestClient(app) as client:
        response = upload(client, "cbc")
    assert response.status_code == 503 and response.json() == {"error": "classifier_unavailable"}
    assert rows(app, Document) == []
    assert store.objects == {}
    assert [(a.operation, a.result) for a in rows(app, AuditLog)] == [("UploadDocument", "classifier_unavailable")]


# --- Review round 1, I2: the provider code itself (e.g. "api:RateLimitError") is kept in the
# log line, sanitised to [A-Za-z0-9_:]{1,80} - not just the bare fact of failure. (The audit
# table itself has no reason column - see write_audit's docstring - so this is log-only, like
# every other reason code.) ---

def test_the_classifier_unavailable_log_line_carries_the_sanitised_provider_code(tmp_path, caplog):
    import logging
    import re as re_module

    class _NoisyUnavailable:
        def classify(self, text):
            raise ClassifierUnavailable("api:RateLimitError: quota <exceeded>! 100%")

        def classify_images(self, images):
            raise ClassifierUnavailable("api:RateLimitError: quota <exceeded>! 100%")

    caplog.set_level(logging.INFO, logger="document-service")
    app, _ = make(tmp_path, classifier=_NoisyUnavailable())
    with TestClient(app) as client:
        upload(client, "cbc")
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "document-service")
    parsed = json.loads(logged)
    assert re_module.fullmatch(r"[A-Za-z0-9_:]{1,80}", parsed["reason"])
    assert parsed["reason"] == "api:RateLimitError:quotaexceeded100"  # unsafe characters dropped, not replaced
    assert "quota <exceeded>" not in logged and "100%" not in logged



def test_the_audit_holds_codes_and_ids_only(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        doc_id = upload(client, "cbc").json()["document_id"]
        client.get("/api/v1/patients/P-10041/documents", headers=H)
    audit = rows(app, AuditLog)
    assert [(a.operation, a.result, a.document_id, a.patient_id) for a in audit] == [
        ("UploadDocument", "ACCEPTED", doc_id, "P-10041"), ("ListDocuments", "ok", None, "P-10041")]


def test_the_log_never_holds_the_patient_id_or_a_file_name(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="document-service")
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        upload(client, "cbc")
    # Only this service's own logger: the test client's httpx logs the request URL itself.
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "document-service")
    assert "P-10041" not in logged and "cbc.pdf" not in logged


# --- Task 2, decision 2: a fixed reason code on every refusal, in the 201 body and the log line -
# never document content, a patient id or a file name. ---

def test_reason_codes_appear_in_the_body_and_the_log_never_with_document_content(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="document-service")
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        expired = upload(client, "cbc", year="2024")
        non_medical = upload(client, "electricity_bill")
        garbage = client.post("/api/v1/patients/P-10041/documents", headers=H,
                              files={"file": ("x.gif", b"GIF89a not a real file, just text", "image/gif")})
    assert expired.json()["reason"] == "too_old"
    assert non_medical.json()["reason"] is None
    assert garbage.status_code == 201 and garbage.json()["reason"] == "not_supported_format"
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "document-service")
    assert '"reason": "too_old"' in logged and '"reason": "not_supported_format"' in logged
    assert "P-10041" not in logged
    assert "cbc.pdf" not in logged and "electricity_bill.pdf" not in logged and "x.gif" not in logged
    assert "GIF89a not a real file" not in logged


def test_an_unsupported_format_upload_is_not_supported_format(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/v1/patients/P-10041/documents", headers=H,
                               files={"file": ("note.txt", b"just some plain text", "text/plain")})
    assert response.status_code == 201
    body = response.json()
    assert body["result"] == "DOCUMENT_UNREADABLE" and body["reason"] == "not_supported_format"
    assert store.objects == {}


# --- Task 2, decision 3: an image, or a scanned PDF with no text layer, is classified through
# vision (FakeClassifier.classify_images); storage.py sets ContentType, main.py sets the object
# key's extension, per kind. ---

def _pil_image_bytes(fmt: str, size: tuple[int, int] = (30, 30)) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(1, 2, 3)).save(buffer, format=fmt)
    return buffer.getvalue()


def _scanned_pdf_bytes() -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (600, 800), color=(9, 9, 9)).save(buffer, format="PDF")
    return buffer.getvalue()


def test_jpeg_and_png_uploads_are_accepted_through_vision_with_content_type_and_extension_per_kind(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        jpeg_body = client.post("/api/v1/patients/P-10041/documents", headers=H,
                                files={"file": ("scan.jpg", _pil_image_bytes("JPEG"), "image/jpeg")}).json()
        png_body = client.post("/api/v1/patients/P-10041/documents", headers=H,
                               files={"file": ("scan.png", _pil_image_bytes("PNG"), "image/png")}).json()
    assert jpeg_body["result"] == "ACCEPTED" and jpeg_body["document_type"] == "CBC"
    assert png_body["result"] == "ACCEPTED" and png_body["document_type"] == "CBC"
    jpeg_key = f"patients/P-10041/{jpeg_body['document_id']}.jpg"
    png_key = f"patients/P-10041/{png_body['document_id']}.png"
    assert list(store.objects) == [jpeg_key, png_key]
    assert store.content_types[jpeg_key] == "image/jpeg"
    assert store.content_types[png_key] == "image/png"


def test_a_pdf_upload_still_gets_the_pdf_extension_and_content_type(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app) as client:
        body = upload(client, "cbc").json()
    key = f"patients/P-10041/{body['document_id']}.pdf"
    assert store.content_types[key] == "application/pdf"


def test_a_scanned_pdf_upload_is_classified_through_vision(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/v1/patients/P-10041/documents", headers=H,
                               files={"file": ("scan.pdf", _scanned_pdf_bytes(), "application/pdf")})
    assert response.status_code == 201
    body = response.json()
    assert body["result"] == "ACCEPTED" and body["document_type"] == "CBC" and body["reason"] is None


def test_the_documents_routes_are_the_whole_api(tmp_path):
    app, _ = make(tmp_path)
    paths = app.openapi()["paths"]
    assert set(paths) == {"/health", "/api/v1/patients/{patient_id}/documents"}
    assert set(paths["/api/v1/patients/{patient_id}/documents"]) == {"get", "post"}


def test_a_database_failure_after_storing_rolls_back_and_says_so(tmp_path, monkeypatch):
    app, store = make(tmp_path)

    def failing_commit(self):
        raise SQLAlchemyError("boom")

    monkeypatch.setattr(Session, "commit", failing_commit)
    with TestClient(app) as client:
        response = upload(client, "cbc")
    assert response.status_code == 503 and response.json() == {"error": "database_unavailable"}
    assert rows(app, Document) == [] and rows(app, AuditLog) == []
    # The put to S3 already happened before the failing commit: the object is orphaned there,
    # unreferenced by any row and never served, exactly as app.main documents.
    assert len(store.objects) == 1


# --- I2: every route's session work is wrapped, so a DB failure anywhere in it becomes a plain
# 503 and never leaks a patient_id through any logger, including sqlalchemy's own. ---

def test_the_engine_hides_bind_parameters_on_error(tmp_path):
    app, _ = make(tmp_path)
    assert app.state.engine.hide_parameters is True


def test_a_database_failure_at_the_duplicate_check_is_503_and_never_logged(tmp_path, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    app, store = make(tmp_path)
    with TestClient(app) as client:
        with app.state.SessionLocal() as session:
            session.execute(text("DROP TABLE documents"))
            session.commit()
        response = upload(client, "cbc")
    assert response.status_code == 503 and response.json() == {"error": "database_unavailable"}
    assert store.objects == {}
    # Only this service's own logger and sqlalchemy's: httpx (the test client) logs the request
    # URL itself, which legitimately carries the patient id in its path.
    assert not any(r.name == "document-service" or r.name.startswith("sqlalchemy") for r in caplog.records)
    logged = "\n".join(r.getMessage() for r in caplog.records
                       if r.name == "document-service" or r.name.startswith("sqlalchemy"))
    assert "P-10041" not in logged


def test_a_database_failure_writing_the_storage_failure_audit_is_503(tmp_path, monkeypatch):
    class Failing:
        def put(self, key, data, content_type="application/pdf"):
            raise StorageFailed("AccessDenied")

    def failing_commit(self):
        raise SQLAlchemyError("boom")

    app, _ = make(tmp_path, store=Failing())
    monkeypatch.setattr(Session, "commit", failing_commit)
    with TestClient(app) as client:
        response = upload(client, "cbc")
    assert response.status_code == 503 and response.json() == {"error": "database_unavailable"}


def test_a_database_failure_at_listing_is_503_and_never_logged(tmp_path, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        with app.state.SessionLocal() as session:
            session.execute(text("DROP TABLE documents"))
            session.commit()
        response = client.get("/api/v1/patients/P-10041/documents", headers=H)
    assert response.status_code == 503 and response.json() == {"error": "database_unavailable"}
    assert not any(r.name == "document-service" or r.name.startswith("sqlalchemy") for r in caplog.records)
    logged = "\n".join(r.getMessage() for r in caplog.records
                       if r.name == "document-service" or r.name.startswith("sqlalchemy"))
    assert "P-10041" not in logged


# --- The auth-before-body ASGI middleware: raw ASGI calls, since FastAPI's own routing (and
# TestClient's convenience layer) would already have parsed the body by the time a route runs. ---

def http_scope(method, path, headers=()):
    return {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "client": ("testclient", 50000), "server": ("testserver", 80), "state": {},
    }


def tracking_receive():
    calls = {"n": 0}

    async def receive():
        calls["n"] += 1
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive, calls


def call_asgi(app, scope):
    receive, calls = tracking_receive()
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m["body"] for m in sent if m["type"] == "http.response.body")
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    return start["status"], body, calls["n"], headers


def test_an_unauthenticated_post_is_refused_before_the_body_is_read(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app):  # runs the lifespan, so the tables exist for the row checks below
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("content-length", "20000000"),
                                    ("content-type", "multipart/form-data; boundary=x")])
        status, body, receive_calls, headers = call_asgi(app, scope)
    assert status == 401 and json.loads(body) == {"error": "unauthorized"}
    assert headers.get("www-authenticate") == "ApiKey"
    assert receive_calls == 0
    assert rows(app, Document) == [] and rows(app, AuditLog) == []
    assert store.objects == {}


def test_a_post_over_the_cap_is_refused_without_reading_the_body(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("x-api-key", KEY), ("content-length", str(MAX_BODY_BYTES + 1)),
                                    ("content-type", "multipart/form-data; boundary=x")])
        status, body, receive_calls, _headers = call_asgi(app, scope)
    assert status == 413 and json.loads(body) == {"error": "too_large"}
    assert receive_calls == 0


def test_a_post_at_the_cap_is_not_refused_for_size(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("x-api-key", KEY), ("content-length", str(MAX_BODY_BYTES)),
                                    ("content-type", "multipart/form-data; boundary=x")])
        status, _body, receive_calls, _headers = call_asgi(app, scope)
    # Passed on size; the route itself then reads an (empty, in this bare scope) body and refuses
    # it as a malformed multipart request - proof this went past the middleware.
    assert status != 413 and receive_calls >= 1


def test_a_post_without_content_length_is_refused_without_reading_the_body(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("x-api-key", KEY), ("content-type", "multipart/form-data; boundary=x")])
        status, body, receive_calls, _headers = call_asgi(app, scope)
    assert status == 411 and json.loads(body) == {"error": "length_required"}
    assert receive_calls == 0


def test_get_and_health_are_unaffected_by_the_content_length_checks(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        get_scope = http_scope("GET", "/api/v1/patients/P-10041/documents", headers=[("x-api-key", KEY)])
        status, body, _receive_calls, _headers = call_asgi(app, get_scope)
        assert status == 200 and json.loads(body) == {"documents": []}

        health_scope = http_scope("GET", "/health")  # no X-API-Key at all
        status, body, _receive_calls, _headers = call_asgi(app, health_scope)
        assert status == 200 and json.loads(body)["status"] == "ok"


# --- Minor: every error shape is {"error": <snake_case code>}, including Starlette's own
# HTTPException (404, 405) and a malformed multipart body (400). ---

def test_an_unknown_route_is_a_json_not_found(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/v1/nope", headers=H)
    assert response.status_code == 404 and response.json() == {"error": "not_found"}


def test_a_disallowed_method_is_a_json_method_not_allowed(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        response = client.delete("/api/v1/patients/P-10041/documents", headers=H)
    assert response.status_code == 405 and response.json() == {"error": "method_not_allowed"}


def test_a_malformed_multipart_body_is_a_json_bad_request(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/patients/P-10041/documents",
            headers={**H, "content-type": "multipart/form-data; boundary=x"},
            content=b"garbage, no boundary markers here")
    assert response.status_code == 400 and response.json() == {"error": "bad_request"}


# --- Minor: uploaded_at is set in Python, and ties in it are broken by insertion order (seq). ---

def test_the_listing_breaks_a_tied_uploaded_at_by_insertion_order(tmp_path, monkeypatch):
    import app.main as main
    fixed = datetime(2026, 9, 22, 10, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(main, "_utc_now", lambda: fixed)
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        first_id = upload(client, "cbc").json()["document_id"]
        second_id = upload(client, "electricity_bill").json()["document_id"]
        docs = client.get("/api/v1/patients/P-10041/documents", headers=H).json()["documents"]
    assert [d["document_id"] for d in docs] == [first_id, second_id]
    assert docs[0]["uploaded_at"] == docs[1]["uploaded_at"]


# --- Sub-project 19 (design D5): every 201 answer carries `llm_usage` - the one classify/vision
# call's tokens, or null when no call was made. The 503 bodies are unchanged. ---

def _openai_classifier(content, usage):
    from types import SimpleNamespace

    from app.classifier import OpenAIClassifier

    def create(**kwargs):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        if usage is not None:
            response.usage = usage
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return OpenAIClassifier("k", "gpt-5.6-luna", client=client)


def _usage(prompt=1234, cached=1000, completion=56):
    from types import SimpleNamespace
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached))


CBC_ANSWER = json.dumps({"is_medical": True, "document_type": "CBC", "document_date": "2026-09-15",
                         "patient_identifier": None})


def test_the_201_answers_full_key_set_includes_llm_usage(tmp_path):
    app, _ = make(tmp_path, classifier=_openai_classifier(CBC_ANSWER, _usage()))
    with TestClient(app) as client:
        body = upload(client, "cbc").json()
    assert body == {"document_id": body["document_id"], "document_type": "CBC", "document_date": "2026-09-15",
                    "result": "ACCEPTED", "reason": None,
                    "llm_usage": {"call": "classify", "model": "gpt-5.6-luna", "input_tokens": 1234,
                                  "cached_input_tokens": 1000, "output_tokens": 56}}


def test_a_duplicates_full_key_set_has_a_null_llm_usage(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        upload(client, "cbc")
        body = upload(client, "cbc").json()
    assert set(body) == {"document_id", "document_type", "document_date", "result", "reason", "duplicate_of",
                         "llm_usage"}
    assert body["result"] == "DUPLICATE_DOCUMENT" and body["llm_usage"] is None


def test_an_image_upload_reports_vision_usage(tmp_path):
    app, _ = make(tmp_path, classifier=_openai_classifier(CBC_ANSWER, _usage(prompt=2000, cached=0, completion=40)))
    with TestClient(app) as client:
        body = client.post("/api/v1/patients/P-10041/documents", headers=H,
                           files={"file": ("scan.png", _pil_image_bytes("PNG"), "image/png")}).json()
    assert body["result"] == "ACCEPTED"
    assert body["llm_usage"] == {"call": "vision", "model": "gpt-5.6-luna", "input_tokens": 2000,
                                 "cached_input_tokens": 0, "output_tokens": 40}


def test_the_fake_classifiers_usage_reaches_the_answer(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        text_body = upload(client, "cbc").json()
        image_body = client.post("/api/v1/patients/P-10041/documents", headers=H,
                                 files={"file": ("scan.jpg", _pil_image_bytes("JPEG"), "image/jpeg")}).json()
    assert (text_body["llm_usage"]["call"], text_body["llm_usage"]["model"]) == ("classify", "fake")
    assert (image_body["llm_usage"]["call"], image_body["llm_usage"]["model"]) == ("vision", "fake")


def test_an_early_rejection_has_a_null_llm_usage(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as client:
        body = client.post("/api/v1/patients/P-10041/documents", headers=H,
                           files={"file": ("x.gif", b"GIF89a not supported", "image/gif")}).json()
    assert body["reason"] == "not_supported_format"
    assert "llm_usage" in body and body["llm_usage"] is None


def test_an_unparsable_answer_is_unreadable_and_keeps_its_usage(tmp_path):
    app, _ = make(tmp_path, classifier=_openai_classifier("not json", _usage()))
    with TestClient(app) as client:
        response = upload(client, "cbc")
    body = response.json()
    assert response.status_code == 201
    assert body["result"] == "DOCUMENT_UNREADABLE" and body["reason"] == "classifier_unparsable"
    assert body["llm_usage"] == {"call": "classify", "model": "gpt-5.6-luna", "input_tokens": 1234,
                                 "cached_input_tokens": 1000, "output_tokens": 56}


def test_a_missing_usage_gives_null_counts_with_the_call_and_model(tmp_path):
    app, _ = make(tmp_path, classifier=_openai_classifier(CBC_ANSWER, None))
    with TestClient(app) as client:
        body = upload(client, "cbc").json()
    assert body["result"] == "ACCEPTED"
    assert body["llm_usage"] == {"call": "classify", "model": "gpt-5.6-luna", "input_tokens": None,
                                 "cached_input_tokens": None, "output_tokens": None}


def test_the_classifier_unavailable_body_is_unchanged(tmp_path):
    app, _ = make(tmp_path, classifier=_UnavailableClassifier())
    with TestClient(app) as client:
        assert upload(client, "cbc").json() == {"error": "classifier_unavailable"}


def test_the_log_line_never_holds_token_counts(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="document-service")
    app, _ = make(tmp_path, classifier=_openai_classifier(CBC_ANSWER, _usage(prompt=987654, completion=45678)))
    with TestClient(app) as client:
        upload(client, "cbc")
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "document-service")
    assert logged and "987654" not in logged and "45678" not in logged and "llm_usage" not in logged
