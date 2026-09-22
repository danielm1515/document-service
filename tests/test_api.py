import re
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.classifier import FakeClassifier
from app.main import create_app
from app.models import AuditLog, Document
from app.storage import InMemoryObjectStore, StorageFailed

FIXTURES = Path(__file__).parent / "fixtures"
KEY = "test-key"
H = {"X-API-Key": KEY}


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
