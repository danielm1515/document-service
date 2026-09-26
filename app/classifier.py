"""Classifies a document: medical or not, which catalog type, its date, and a patient identifier
if it shows one (HospitalAgent design §4.3). Classification only - the prompt forbids
interpreting a result. The input is either extracted text (`classify`) or, for an image upload or
a scanned PDF with no text layer, one or more page images (`classify_images`) - same prompt
contract, same schema, one call either way.

Any unusable answer raises ClassifierFailed, which the intake turns into DOCUMENT_UNREADABLE
(fail closed, reason `classifier_unparsable`). A provider failure (rate limit, quota, timeout,
connection, auth - anything the SDK raises as an `OpenAIError`) is not a verdict on the file: it
raises ClassifierUnavailable instead, which `app/main.py` turns into `503 classifier_unavailable`,
the same pattern as a storage failure.

Every answer that arrived carries the call's token usage (`LLMUsage`, Sub-project 19) - on the
Classification, or on the ClassifierFailed of an unusable answer. A provider failure carries none."""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Protocol

from .catalog import CODES
from .magic import mime_of

MAX_TEXT_CHARS = 8000
MAX_IMAGES = 4

PROMPT = (
    "You classify one document for a hospital's intake system. The input is either text extracted "
    "from a PDF, or one or more images of the document (a photo, or a page from a scanned PDF with "
    "no text layer) - classify what the image(s) show exactly as you would the extracted text, using "
    "the same fields below. The text, when given, is usually Hebrew. Answer only with the JSON the "
    "schema allows.\n"
    "- is_medical: true only for a clinical document (lab results, a test report, a medical summary); "
    "false for bills, receipts, letters and anything else.\n"
    f"- document_type: one of {', '.join(CODES)} when the document is exactly that type, otherwise null. "
    "CBC = complete blood count; COAGULATION_TESTS = coagulation / PT / INR tests; ECG = electrocardiogram; "
    "URINALYSIS = urine test; PREOP_SUMMARY = a pre-operative assessment summary. A pre-operative summary "
    "that lists other tests is still PREOP_SUMMARY.\n"
    "- document_date: the date the document was issued (e.g. 'תאריך הפקה'), as YYYY-MM-DD, or null.\n"
    "- patient_identifier: an explicit patient identifier printed in the document, or null. Never a "
    "document, invoice or customer number.\n"
    "Never interpret results, never judge whether a value is normal, and never give medical advice."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["is_medical", "document_type", "document_date", "patient_identifier"],
    "properties": {
        "is_medical": {"type": "boolean"},
        "document_type": {"anyOf": [{"type": "string", "enum": list(CODES)}, {"type": "null"}]},
        "document_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "patient_identifier": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}


@dataclass(frozen=True)
class LLMUsage:
    """The token usage of the one classify/vision call behind an answer (Sub-project 19, D1/D5) -
    counts and codes only, never text. `call` is "classify" (extracted text) or "vision" (page
    images). The three counts are all None when the response's `usage` was missing or malformed:
    never a guess, but the call and the model still say that a call was billed."""
    call: str
    model: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None

    def as_json(self) -> dict[str, Any]:
        return {"call": self.call, "model": self.model, "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens, "output_tokens": self.output_tokens}


def _count(value: Any) -> int | None:
    """A non-negative int, or None (a bool is not a count)."""
    return value if type(value) is int and value >= 0 else None


def _usage_of(response: Any, call: str, model: str) -> LLMUsage:
    """Reads `prompt_tokens`, `prompt_tokens_details.cached_tokens` (absent means 0) and
    `completion_tokens`. Anything missing or malformed - a non-int, a negative, cached > prompt -
    gives null counts for all three rather than a partial or guessed figure."""
    none = LLMUsage(call, model, None, None, None)
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return none
        prompt = _count(getattr(usage, "prompt_tokens", None))
        completion = _count(getattr(usage, "completion_tokens", None))
        details = getattr(usage, "prompt_tokens_details", None)
        raw_cached = getattr(details, "cached_tokens", None) if details is not None else None
        cached = 0 if raw_cached is None else _count(raw_cached)
    except Exception:  # an odd response object is bookkeeping's problem, never the upload's
        return none
    if prompt is None or completion is None or cached is None or cached > prompt:
        return none
    return LLMUsage(call, model, prompt, cached, completion)


class ClassifierFailed(Exception):
    """No usable classification. The reason is a code, never the document's text. `usage` is the
    usage of an answer that arrived but could not be used (those tokens were billed); None when
    no answer arrived."""

    def __init__(self, reason: str, *, usage: LLMUsage | None = None) -> None:
        super().__init__(reason)
        self.usage = usage


class ClassifierUnavailable(Exception):
    """The provider itself failed (rate limit, quota, timeout, connection, auth - `api:<type>`).
    Never a verdict on the file: the caller must not treat this as DOCUMENT_UNREADABLE. Carries no
    usage - an API error bills nothing to report."""


@dataclass(frozen=True)
class Classification:
    is_medical: bool
    document_type: str | None
    document_date: date | None
    patient_identifier: str | None
    # The usage of the call that produced this answer (Sub-project 19). Bookkeeping, not part of
    # the verdict, so it takes no part in equality.
    usage: LLMUsage | None = field(default=None, compare=False)


class Classifier(Protocol):
    def classify(self, text: str) -> Classification: ...
    def classify_images(self, images: list[bytes]) -> Classification: ...


def _parse(data: Any) -> Classification:
    if not isinstance(data, dict) or set(data) != set(SCHEMA["required"]):
        raise ClassifierFailed("shape")
    is_medical, doc_type, raw_date, identifier = (data["is_medical"], data["document_type"],
                                                  data["document_date"], data["patient_identifier"])
    if not isinstance(is_medical, bool):
        raise ClassifierFailed("is_medical")
    if doc_type is not None and doc_type not in CODES:
        raise ClassifierFailed("document_type")
    if raw_date is not None:
        if not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            raise ClassifierFailed("document_date")
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError:
            raise ClassifierFailed("document_date") from None
    else:
        parsed_date = None
    if identifier is not None and not isinstance(identifier, str):
        raise ClassifierFailed("patient_identifier")
    return Classification(is_medical, doc_type, parsed_date, identifier or None)


class OpenAIClassifier:
    """The same call the Hospital Agent makes: Chat Completions, reasoning_effort="none", a strict
    JSON Schema. The client is built lazily; tests inject one."""

    def __init__(self, api_key: str, model: str, *, client: Any = None, timeout_seconds: float = 30.0) -> None:
        self.model, self._api_key, self._timeout, self._client = model, api_key, timeout_seconds, client

    def __repr__(self) -> str:
        return f"OpenAIClassifier(model={self.model!r})"  # never the key

    def _openai(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout, max_retries=0)
        return self._client

    def classify(self, text: str) -> Classification:
        return self._call("classify", [{"role": "system", "content": PROMPT},
                                       {"role": "user", "content": text[:MAX_TEXT_CHARS]}])

    def classify_images(self, images: list[bytes]) -> Classification:
        """Sends up to MAX_IMAGES page images as `image_url` data-URL parts, same prompt contract,
        same schema, one call - the image goes to the same processor the extracted text already
        goes to (README, privacy)."""
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Classify the document shown in the following image(s)."}]
        for image in images[:MAX_IMAGES]:
            b64 = base64.b64encode(image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_of(image)};base64,{b64}"}})
        return self._call("vision", [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}])

    def _call(self, call: str, messages: list[dict[str, Any]]) -> Classification:
        """`call` names the usage ("classify" or "vision"). An answer that arrived carries its
        usage whether it was usable or not - an unusable one on its ClassifierFailed."""
        import openai

        response = None
        try:
            response = self._openai().chat.completions.create(
                model=self.model,
                reasoning_effort="none",
                messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "document_classification", "strict": True, "schema": SCHEMA}},
            )
            data = json.loads(response.choices[0].message.content)
        except openai.OpenAIError as exc:
            raise ClassifierUnavailable(f"api:{type(exc).__name__}") from None
        except (json.JSONDecodeError, TypeError, IndexError, AttributeError):
            usage = _usage_of(response, call, self.model) if response is not None else None
            raise ClassifierFailed("unparsable", usage=usage) from None
        usage = _usage_of(response, call, self.model)
        try:
            found = _parse(data)
        except ClassifierFailed as exc:
            raise ClassifierFailed(str(exc), usage=usage) from None
        return replace(found, usage=usage)


# The fake: deterministic keyword rules over the demo files' Hebrew. Order matters - a
# pre-operative summary lists the other tests, and the urine test mentions "טרום-ניתוח".
_RULES: tuple[tuple[str, str], ...] = (
    ("סיכום הערכה", "PREOP_SUMMARY"),
    ("בדיקת שתן", "URINALYSIS"),
    ("חשמלית של הלב", "ECG"),
    ("בדיקות קרישה", "COAGULATION_TESTS"),
    ("ספירת דם", "CBC"),
)
_NON_MEDICAL = ("שאינו רפואי", "לא מסמך רפואי", "חשבון חשמל")


class FakeClassifier:
    """For the tests and offline runs only. Reports a fixed, deterministic usage under the model
    name "fake" (no real call is made, so no real count exists; the figures only prove the usage
    path end to end)."""

    model = "fake"
    TEXT_USAGE = LLMUsage("classify", "fake", 900, 0, 40)
    IMAGE_USAGE = LLMUsage("vision", "fake", 1200, 0, 40)

    def classify(self, text: str) -> Classification:
        if not text.strip():
            raise ClassifierFailed("empty")
        found = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", text)
        doc_date = date(int(found.group(3)), int(found.group(2)), int(found.group(1))) if found else None
        identifier = re.search(r"\bP-\d{4,}\b", text)
        patient = identifier.group(0) if identifier else None
        usage = self.TEXT_USAGE
        if any(word in text for word in _NON_MEDICAL):
            return Classification(False, None, doc_date, patient, usage)
        for word, code in _RULES:
            if word in text:
                return Classification(True, code, doc_date, patient, usage)
        return Classification(True, None, doc_date, patient, usage)

    def classify_images(self, images: list[bytes]) -> Classification:
        """There is no OCR here to run a keyword rule against, so this deterministically reads the
        document as a CBC dated 2026-09-15 (the demo files' own date) - enough for a test to prove
        an image or a scanned PDF actually reached this method, without pretending to read pixels."""
        if not images:
            raise ClassifierFailed("empty")
        return Classification(True, "CBC", date(2026, 9, 15), None, self.IMAGE_USAGE)
