import base64
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pypdf
import pytest

from app.classifier import (MAX_IMAGES, MAX_TEXT_CHARS, PROMPT, SCHEMA, Classification, ClassifierFailed,
                            ClassifierUnavailable, FakeClassifier, LLMUsage, OpenAIClassifier)

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


def test_a_400_bad_request_is_also_unavailable_not_unparsable():
    """Review round 1, M4: a model that rejects image input (e.g. OPENAI_MODEL pointed at a
    non-vision model) answers 400, which is still an openai.OpenAIError subclass - so it is a
    provider failure (classifier_unavailable), never DOCUMENT_UNREADABLE."""
    import httpx
    import openai
    request = httpx.Request("POST", "https://api.openai.com")
    response = httpx.Response(400, request=request, json={"error": {"message": "no image support"}})
    error = openai.BadRequestError("no image support", response=response, body=None)
    classifier, _ = openai_with(raises=error)
    with pytest.raises(ClassifierUnavailable):
        classifier.classify_images([b"\xff\xd8\xff"])


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


# --- Sub-project 19 (design D1/D5): the one call's token usage travels with its answer. ---

def usage_of(prompt=1234, cached=1000, completion=56, details=True):
    fields = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": 0}
    if details:
        fields["prompt_tokens_details"] = SimpleNamespace(cached_tokens=cached)
    return SimpleNamespace(**fields)


class UsageCompletions(FakeCompletions):
    def __init__(self, content=None, usage=None, omit_usage=False):
        super().__init__(content)
        self.usage, self.omit_usage = usage, omit_usage

    def create(self, **kwargs):
        response = super().create(**kwargs)
        if not self.omit_usage:
            response.usage = self.usage
        return response


def openai_reporting(content=None, usage=None, omit_usage=False):
    completions = UsageCompletions(content, usage, omit_usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIClassifier("k", "gpt-5.6-luna", client=client)


def test_a_text_classification_carries_the_calls_usage():
    got = openai_reporting(answer(), usage_of()).classify("text")
    assert got.usage == LLMUsage("classify", "gpt-5.6-luna", 1234, 1000, 56)
    assert got.usage.as_json() == {"call": "classify", "model": "gpt-5.6-luna", "input_tokens": 1234,
                                   "cached_input_tokens": 1000, "output_tokens": 56}


def test_an_image_classification_carries_the_calls_usage_as_vision():
    got = openai_reporting(answer(), usage_of(prompt=2000, cached=0, completion=40)).classify_images([b"\xff\xd8\xff"])
    assert got.usage == LLMUsage("vision", "gpt-5.6-luna", 2000, 0, 40)


@pytest.mark.parametrize("details", [False, SimpleNamespace(cached_tokens=None), None, SimpleNamespace()])
def test_absent_cached_tokens_count_as_zero(details):
    usage = usage_of(prompt=300, completion=20, details=False)
    if details is not False:
        usage.prompt_tokens_details = details
    got = openai_reporting(answer(), usage).classify("text")
    assert got.usage == LLMUsage("classify", "gpt-5.6-luna", 300, 0, 20)


@pytest.mark.parametrize("omit_usage", [True, False])
def test_a_missing_usage_keeps_the_call_and_model_with_null_counts(omit_usage):
    got = openai_reporting(answer(), None, omit_usage=omit_usage).classify("text")
    assert got.usage == LLMUsage("classify", "gpt-5.6-luna", None, None, None)
    assert got.document_type == "CBC"  # bookkeeping never changes the verdict


@pytest.mark.parametrize("usage", [
    usage_of(prompt="1234"), usage_of(prompt=-1), usage_of(prompt=True, cached=0), usage_of(prompt=None),
    usage_of(completion=None), usage_of(completion=2.5), usage_of(completion=-3), usage_of(completion=True),
    usage_of(cached="5"), usage_of(cached=-1), usage_of(prompt=10, cached=11),
    {"prompt_tokens": 1, "completion_tokens": 1},
])
def test_a_malformed_usage_gives_null_counts_never_a_guess(usage):
    got = openai_reporting(answer(), usage).classify("text")
    assert got.usage == LLMUsage("classify", "gpt-5.6-luna", None, None, None)


@pytest.mark.parametrize("content", ["not json", answer(document_type="ELECTRICITY"), json.dumps({"is_medical": True})])
def test_an_unusable_answer_still_carries_the_usage_it_was_billed(content):
    with pytest.raises(ClassifierFailed) as caught:
        openai_reporting(content, usage_of()).classify("text")
    assert caught.value.usage == LLMUsage("classify", "gpt-5.6-luna", 1234, 1000, 56)


def test_an_unusable_image_answer_carries_vision_usage():
    with pytest.raises(ClassifierFailed) as caught:
        openai_reporting("not json", usage_of()).classify_images([b"\xff\xd8\xff"])
    assert caught.value.usage == LLMUsage("vision", "gpt-5.6-luna", 1234, 1000, 56)


def test_an_answer_with_no_choices_is_unparsable_with_its_usage():
    classifier = openai_reporting(answer(), usage_of())
    classifier._client.chat.completions.create = lambda **kwargs: SimpleNamespace(choices=[], usage=usage_of())
    with pytest.raises(ClassifierFailed) as caught:
        classifier.classify("text")
    assert caught.value.usage == LLMUsage("classify", "gpt-5.6-luna", 1234, 1000, 56)


def test_a_classifier_failed_without_an_answer_has_no_usage():
    assert ClassifierFailed("empty").usage is None


def test_the_fake_reports_deterministic_usage_for_each_call():
    fake = FakeClassifier()
    text_usage = fake.classify(text_of("2026", "cbc")).usage
    image_usage = fake.classify_images([b"\xff\xd8\xff"]).usage
    # every branch of the fake's rules: a catalog type, no type, and not medical
    assert text_usage == fake.classify("ספירת דם 01.09.2026").usage
    assert text_usage == fake.classify("מסמך כלשהו 01.09.2026").usage
    assert text_usage == fake.classify(text_of("2026", "electricity_bill")).usage
    assert (text_usage.call, text_usage.model) == ("classify", "fake")
    assert (image_usage.call, image_usage.model) == ("vision", "fake")
    for usage in (text_usage, image_usage):
        assert all(isinstance(n, int) and n >= 0 for n in
                   (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens))
        assert usage.cached_input_tokens <= usage.input_tokens
