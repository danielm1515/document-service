"""Classifies a document: medical or not, which catalog type, its date, and a patient identifier
if it shows one (HospitalAgent design §4.3). Classification only - the prompt forbids
interpreting a result. The input is either extracted text (`classify`) or, for an image upload or
a scanned PDF with no text layer, one or more page images (`classify_images`) - same prompt
contract, same schema, one call either way.

Any unusable answer raises ClassifierFailed, which the intake turns into DOCUMENT_UNREADABLE
(fail closed, reason `classifier_unparsable`). A provider failure (rate limit, quota, timeout,
connection, auth - anything the SDK raises as an `OpenAIError`) is not a verdict on the file: it
raises ClassifierUnavailable instead, which `app/main.py` turns into `503 classifier_unavailable`,
the same pattern as a storage failure."""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
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


class ClassifierFailed(Exception):
    """No usable classification. The reason is a code, never the document's text."""


class ClassifierUnavailable(Exception):
    """The provider itself failed (rate limit, quota, timeout, connection, auth - `api:<type>`).
    Never a verdict on the file: the caller must not treat this as DOCUMENT_UNREADABLE."""


@dataclass(frozen=True)
class Classification:
    is_medical: bool
    document_type: str | None
    document_date: date | None
    patient_identifier: str | None


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
        return self._call([{"role": "system", "content": PROMPT},
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
        return self._call([{"role": "system", "content": PROMPT}, {"role": "user", "content": content}])

    def _call(self, messages: list[dict[str, Any]]) -> Classification:
        import openai

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
            raise ClassifierFailed("unparsable") from None
        return _parse(data)


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
    """For the tests and offline runs only."""

    def classify(self, text: str) -> Classification:
        if not text.strip():
            raise ClassifierFailed("empty")
        found = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", text)
        doc_date = date(int(found.group(3)), int(found.group(2)), int(found.group(1))) if found else None
        identifier = re.search(r"\bP-\d{4,}\b", text)
        patient = identifier.group(0) if identifier else None
        if any(word in text for word in _NON_MEDICAL):
            return Classification(False, None, doc_date, patient)
        for word, code in _RULES:
            if word in text:
                return Classification(True, code, doc_date, patient)
        return Classification(True, None, doc_date, patient)

    def classify_images(self, images: list[bytes]) -> Classification:
        """There is no OCR here to run a keyword rule against, so this deterministically reads the
        document as a CBC dated 2026-09-15 (the demo files' own date) - enough for a test to prove
        an image or a scanned PDF actually reached this method, without pretending to read pixels."""
        if not images:
            raise ClassifierFailed("empty")
        return Classification(True, "CBC", date(2026, 9, 15), None)
