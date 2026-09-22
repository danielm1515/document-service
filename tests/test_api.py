import asyncio
import json
import re
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.classifier import FakeClassifier
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
                    "document_date": "2026-09-15", "result": "ACCEPTED"}
    assert list(store.objects) == [f"patients/P-10041/{body['document_id']}.pdf"]
    assert store.objects[f"patients/P-10041/{body['document_id']}.pdf"] == pdf("2026", "cbc")
    [doc] = listing["documents"]
    assert {k: doc[k] for k in ("document_id", "document_type", "document_date", "result")} == {
        "document_id": body["document_id"], "document_type": "CBC", "document_date": "2026-09-15", "result": "ACCEPTED"}
    assert "uploaded_at" in doc


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
        def put(self, key, data):
            raise StorageFailed("AccessDenied")
    app, _ = make(tmp_path, store=Failing())
    with TestClient(app) as client:
        response = upload(client, "cbc")
    assert response.status_code == 503 and response.json() == {"error": "storage_unavailable"}
    assert rows(app, Document) == []
    assert [(a.operation, a.result) for a in rows(app, AuditLog)] == [("UploadDocument", "storage_unavailable")]


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
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m["body"] for m in sent if m["type"] == "http.response.body")
    return status, body, calls["n"]


def test_an_unauthenticated_post_is_refused_before_the_body_is_read(tmp_path):
    app, store = make(tmp_path)
    with TestClient(app):  # runs the lifespan, so the tables exist for the row checks below
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("content-length", "20000000"),
                                    ("content-type", "multipart/form-data; boundary=x")])
        status, body, receive_calls = call_asgi(app, scope)
    assert status == 401 and json.loads(body) == {"error": "unauthorized"}
    assert receive_calls == 0
    assert rows(app, Document) == [] and rows(app, AuditLog) == []
    assert store.objects == {}


def test_a_post_over_the_cap_is_refused_without_reading_the_body(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("x-api-key", KEY), ("content-length", str(MAX_BODY_BYTES + 1)),
                                    ("content-type", "multipart/form-data; boundary=x")])
        status, body, receive_calls = call_asgi(app, scope)
    assert status == 413 and json.loads(body) == {"error": "too_large"}
    assert receive_calls == 0


def test_a_post_at_the_cap_is_not_refused_for_size(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("x-api-key", KEY), ("content-length", str(MAX_BODY_BYTES)),
                                    ("content-type", "multipart/form-data; boundary=x")])
        status, _body, receive_calls = call_asgi(app, scope)
    # Passed on size; the route itself then reads an (empty, in this bare scope) body and refuses
    # it as an incomplete multipart request - proof this went past the middleware.
    assert status != 413 and receive_calls >= 1


def test_a_post_without_content_length_is_refused_without_reading_the_body(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        scope = http_scope("POST", "/api/v1/patients/P-10041/documents",
                           headers=[("x-api-key", KEY), ("content-type", "multipart/form-data; boundary=x")])
        status, body, receive_calls = call_asgi(app, scope)
    assert status == 411 and json.loads(body) == {"error": "length_required"}
    assert receive_calls == 0


def test_get_and_health_are_unaffected_by_the_content_length_checks(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app):
        get_scope = http_scope("GET", "/api/v1/patients/P-10041/documents", headers=[("x-api-key", KEY)])
        status, body, _ = call_asgi(app, get_scope)
        assert status == 200 and json.loads(body) == {"documents": []}

        health_scope = http_scope("GET", "/health")  # no X-API-Key at all
        status, body, _ = call_asgi(app, health_scope)
        assert status == 200 and json.loads(body)["status"] == "ok"
