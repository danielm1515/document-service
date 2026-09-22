from fastapi.testclient import TestClient

from app import catalog
from app.main import create_app


def make(tmp_path, **kwargs):
    return create_app(f"sqlite:///{(tmp_path / 'd.db').as_posix()}", **kwargs)


def test_the_catalog_is_the_designs_table():
    assert [(t.code, t.label, t.max_age_days) for t in catalog.DOCUMENT_TYPES] == [
        ("CBC", "ספירת דם מלאה", 90), ("COAGULATION_TESTS", "בדיקות קרישה", 90),
        ("ECG", "תרשים פעילות חשמלית של הלב", 180), ("URINALYSIS", "בדיקת שתן", 90),
        ("PREOP_SUMMARY", "סיכום טרום ניתוח", 30)]


def test_health_says_what_is_configured(tmp_path):
    with TestClient(make(tmp_path, classifier=None, store=None)) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "database": "ok", "classifier": "not_configured", "storage": "not_configured"}


def test_health_needs_no_key(tmp_path):
    with TestClient(make(tmp_path, api_key="k")) as client:
        assert client.get("/health").status_code == 200


def test_without_a_configured_key_nothing_is_authorised(tmp_path):
    app = make(tmp_path, api_key="")
    with TestClient(app):
        class R:  # the minimal request shape authorised() reads
            headers = {"X-API-Key": ""}
        assert app.state.authorised(R()) is False
        R.headers = {"X-API-Key": "anything"}
        assert app.state.authorised(R()) is False


def test_the_right_key_is_authorised_and_a_wrong_one_is_not(tmp_path):
    app = make(tmp_path, api_key="secret-key")
    with TestClient(app):
        class R:
            headers = {"X-API-Key": "secret-key"}
        assert app.state.authorised(R()) is True
        R.headers = {"X-API-Key": "nope"}
        assert app.state.authorised(R()) is False


def test_the_classifier_comes_from_the_environment(tmp_path, monkeypatch):
    from app.classifier import OpenAIClassifier
    assert make(tmp_path).state.classifier is None           # conftest cleared OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    built = make(tmp_path).state.classifier
    assert isinstance(built, OpenAIClassifier) and "sk-test" not in repr(built)


def test_the_store_comes_from_the_environment(tmp_path, monkeypatch):
    from app.storage import S3ObjectStore
    assert make(tmp_path).state.store is None
    monkeypatch.setenv("S3_BUCKET", "hospital-docs-test")
    built = make(tmp_path).state.store
    assert isinstance(built, S3ObjectStore) and built.bucket == "hospital-docs-test" and built.region == "eu-north-1"
