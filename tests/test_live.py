"""Live checks - the only tests that reach the network, and only when asked (README, "בדיקות")."""
import os
from pathlib import Path

import pypdf
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
live_llm = pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call OpenAI")
live_s3 = pytest.mark.skipif(os.getenv("RUN_LIVE_S3") != "1", reason="set RUN_LIVE_S3=1 to write to the bucket")


@live_llm
@pytest.mark.parametrize("name, expected", [("cbc", "CBC"), ("coagulation", "COAGULATION_TESTS"), ("ecg", "ECG"),
                                            ("urinalysis", "URINALYSIS"), ("preop_summary", "PREOP_SUMMARY"),
                                            ("electricity_bill", None)])
def test_the_real_model_classifies_the_demo_files(name, expected):
    from app.classifier import OpenAIClassifier
    text = "".join(p.extract_text() or "" for p in pypdf.PdfReader(FIXTURES / "2026" / f"{name}.pdf").pages)
    got = OpenAIClassifier(os.environ["LIVE_OPENAI_API_KEY"], os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")).classify(text)
    assert got.document_type == expected and got.is_medical == (expected is not None)


@live_s3
def test_the_real_bucket_accepts_a_private_encrypted_put():
    from app.storage import S3ObjectStore
    S3ObjectStore(os.environ["LIVE_S3_BUCKET"], os.environ.get("AWS_REGION", "eu-north-1")).put(
        "patients/LIVE-TEST/DOC-LIVE.pdf", (FIXTURES / "2026" / "cbc.pdf").read_bytes())
