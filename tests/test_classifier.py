import base64
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pypdf
import pytest

from app.classifier import (MAX_IMAGES, MAX_TEXT_CHARS, PROMPT, SCHEMA, Classification, ClassifierFailed,
                            ClassifierUnavailable, FakeClassifier, OpenAIClassifier)

FIXTURES = Path(__file__).parent / "fixtures"


def text_of(year, name):
    return "".join(p.extract_text() or "" for p in pypdf.PdfReader(FIXTURES / year / f"{name}.pdf").pages)


@pytest.mark.parametrize("name, expected", [
    ("cbc", ("CBC", True)), ("coagulation", ("COAGULATION_TESTS", True)), ("ecg", ("ECG", True)),
    ("urinalysis", ("URINALYSIS", True)), ("preop_summary", ("PREOP_SUMMARY", True)),
    ("electricity_bill", (None, False)),
])
@pytest.mark.parametrize("year", ["2024", "2026"])
def test_the_fake_classifies_every_demo_file(year, name, expected):
    got = FakeClassifier().classify(text_of(year, name))
    assert (got.document_type, got.is_medical) == expected
    assert got.document_date == (date(2024, 8, 12) if year == "2024" else date(2026, 9, 15))
    assert got.patient_identifier is None


def test_the_preop_summary_is_not_mistaken_for_the_tests_it_lists():
    """It names CBC, coagulation and ECG among the documents it checked - still PREOP_SUMMARY."""
    assert FakeClassifier().classify(text_of("2026", "preop_summary")).document_type == "PREOP_SUMMARY"


def test_the_fake_reads_a_patient_identifier_of_the_idp_shape():
    got = FakeClassifier().classify("ספירת דם מלאה מסמך רפואי 01.09.2026 מטופל P-20000")
    assert got.patient_identifier == "P-20000"


def test_the_fake_fails_on_empty_text():
    with pytest.raises(ClassifierFailed):
        FakeClassifier().classify("   ")


def test_the_fake_classifies_images_deterministically():
    """There is no OCR in the fake, so it always answers a dated CBC - enough to prove the image
    path was actually reached, without pretending to read pixels."""
    got = FakeClassifier().classify_images([b"\xff\xd8\xffsome-jpeg-bytes"])
    assert got == Classification(True, "CBC", date(2026, 9, 15), None)


def test_the_fake_fails_on_no_images():
    with pytest.raises(ClassifierFailed):
        FakeClassifier().classify_images([])


class FakeCompletions:
    def __init__(self, content=None, raises=None):
        self.content, self.raises, self.calls = content, raises, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def openai_with(content=None, raises=None):
    completions = FakeCompletions(content, raises)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIClassifier("k", "gpt-5.6-luna", client=client), completions


def answer(**overrides):
    base = {"is_medical": True, "document_type": "CBC", "document_date": "2026-09-15", "patient_identifier": None}
    return json.dumps({**base, **overrides})


def test_openai_call_shape():
    classifier, completions = openai_with(answer())
    result = classifier.classify("x" * (MAX_TEXT_CHARS + 500))
    assert result == Classification(True, "CBC", date(2026, 9, 15), None)
    [call] = completions.calls
    assert call["model"] == "gpt-5.6-luna" and call["reasoning_effort"] == "none"
    assert call["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "document_classification", "strict": True, "schema": SCHEMA}}
    assert call["messages"][0] == {"role": "system", "content": PROMPT}
    assert len(call["messages"][1]["content"]) == MAX_TEXT_CHARS  # only the capped extracted text is sent


@pytest.mark.parametrize("content", [
    "not json", "[]", answer(document_type="ELECTRICITY"), answer(is_medical="yes"),
    answer(document_date="15/09/2026"), json.dumps({"is_medical": True}),
])
def test_an_unusable_answer_fails(content):
    classifier, _ = openai_with(content)
    with pytest.raises(ClassifierFailed):
        classifier.classify("text")


def test_an_api_error_is_unavailable_not_unparsable():
    """A provider failure is never a verdict on the file (Task 2, decision 1)."""
    import httpx
    import openai
    classifier, _ = openai_with(raises=openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")))
    with pytest.raises(ClassifierUnavailable):
        classifier.classify("text")


def test_openai_classify_images_call_shape():
    """Same prompt contract, same schema, one call - `image_url` data-URL parts in the user
    message instead of the extracted text (Task 2, decision 3)."""
    classifier, completions = openai_with(answer())
    result = classifier.classify_images([b"\xff\xd8\xffJPEGDATA", b"\x89PNG\r\n\x1a\nPNGDATA"])
    assert result == Classification(True, "CBC", date(2026, 9, 15), None)
    [call] = completions.calls
    assert call["model"] == "gpt-5.6-luna" and call["reasoning_effort"] == "none"
    assert call["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "document_classification", "strict": True, "schema": SCHEMA}}
    assert call["messages"][0] == {"role": "system", "content": PROMPT}
    user_content = call["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1] == {"type": "image_url",
                               "image_url": {"url": "data:image/jpeg;base64,"
                                             + base64.b64encode(b"\xff\xd8\xffJPEGDATA").decode()}}
    assert user_content[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_classify_images_sends_at_most_max_images():
    classifier, completions = openai_with(answer())
    classifier.classify_images([b"\xff\xd8\xff"] * (MAX_IMAGES + 5))
    [call] = completions.calls
    assert len(call["messages"][1]["content"]) == 1 + MAX_IMAGES  # 1 text part + at most MAX_IMAGES images


def test_classify_images_api_error_is_unavailable():
    import httpx
    import openai
    classifier, _ = openai_with(raises=openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")))
    with pytest.raises(ClassifierUnavailable):
        classifier.classify_images([b"\xff\xd8\xff"])


def test_classify_images_an_unusable_answer_is_classifier_failed():
    classifier, _ = openai_with("not json")
    with pytest.raises(ClassifierFailed):
        classifier.classify_images([b"\xff\xd8\xff"])


def test_the_prompt_describes_the_image_case():
    assert "image" in PROMPT.lower()


def test_a_null_date_and_type_are_allowed_answers():
    classifier, _ = openai_with(answer(is_medical=False, document_type=None, document_date=None))
    assert classifier.classify("text") == Classification(False, None, None, None)


def test_the_prompt_asks_for_classification_only():
    assert "never interpret" in PROMPT.lower()
