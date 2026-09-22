"""The tests never reach OpenAI or AWS: a shell that has the real variables set must not turn a
test into a live call, so they are cleared before every test. A test that wants a port passes it."""
import pytest

LIVE_VARIABLES = ("OPENAI_API_KEY", "S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                  "AWS_SESSION_TOKEN", "AWS_PROFILE", "DOCUMENT_API_KEY")


@pytest.fixture(autouse=True)
def no_live_configuration(monkeypatch):
    for name in LIVE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
