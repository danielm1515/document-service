"""Live checks - the only tests that reach the network, and only when asked (README, "בדיקות")."""
import os
from datetime import date
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
live_llm = pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call OpenAI")
live_s3 = pytest.mark.skipif(os.getenv("RUN_LIVE_S3") != "1", reason="set RUN_LIVE_S3=1 to write to the bucket")


@live_llm
@pytest.mark.parametrize("name, expected_type, expected_result", [
    ("cbc", "CBC", "ACCEPTED"), ("coagulation", "COAGULATION_TESTS", "ACCEPTED"), ("ecg", "ECG", "ACCEPTED"),
    ("urinalysis", "URINALYSIS", "ACCEPTED"), ("preop_summary", "PREOP_SUMMARY", "ACCEPTED"),
    ("electricity_bill", None, "NON_MEDICAL_DOCUMENT"),
])
def test_the_real_model_runs_intake_on_the_demo_files(name, expected_type, expected_result):
    """I5: the whole intake outcome, through the real classifier - not just is_medical/document_type."""
    from app.classifier import OpenAIClassifier
    from app.intake import ACCEPTED, run_intake
    data = (FIXTURES / "2026" / f"{name}.pdf").read_bytes()
    classifier = OpenAIClassifier(os.environ["LIVE_OPENAI_API_KEY"], os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"))
    outcome = run_intake(data, "P-10041", classifier=classifier, today=date(2026, 9, 22),
                         is_duplicate=lambda sha: None, max_bytes=10 * 1024 * 1024)
    assert outcome.result == expected_result and outcome.document_type == expected_type
    if expected_result == ACCEPTED:
        assert outcome.document_date == date(2026, 9, 15)


@live_s3
def test_the_real_bucket_accepts_a_private_encrypted_put():
    """I1: built from LIVE_AWS_* explicitly - conftest clears AWS_ACCESS_KEY_ID/SECRET for every
    other test, so boto3's default credential chain would find nothing here either."""
    import boto3
    from app.storage import S3ObjectStore
    region = os.environ.get("AWS_REGION", "eu-north-1")
    client = boto3.client("s3", region_name=region,
                          aws_access_key_id=os.environ["LIVE_AWS_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["LIVE_AWS_SECRET_ACCESS_KEY"])
    S3ObjectStore(os.environ["LIVE_S3_BUCKET"], region, client=client).put(
        "patients/LIVE-TEST/DOC-LIVE.pdf", (FIXTURES / "2026" / "cbc.pdf").read_bytes())
