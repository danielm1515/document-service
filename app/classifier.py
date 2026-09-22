"""Classifies a document's extracted text: medical or not, which catalog type, its date, and a
patient identifier if the text shows one (HospitalAgent design §4.3). Classification only - the
prompt forbids interpreting a result. Any unusable answer raises ClassifierFailed, which the
intake turns into DOCUMENT_UNREADABLE (fail closed)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from .catalog import CODES

MAX_TEXT_CHARS = 8000

PROMPT = (
    "You classify one document for a hospital's intake system. The text was extracted from a PDF "
    "and is usually Hebrew. Answer only with the JSON the schema allows.\n"
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


@dataclass(frozen=True)
class Classification:
    is_medical: bool
    document_type: str | None
    document_date: date | None
    patient_identifier: str | None


class Classifier(Protocol):
    def classify(self, text: str) -> Classification: ...


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
        import openai

        try:
            response = self._openai().chat.completions.create(
                model=self.model,
                reasoning_effort="none",
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": text[:MAX_TEXT_CHARS]}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "document_classification", "strict": True, "schema": SCHEMA}},
            )
            data = json.loads(response.choices[0].message.content)
        except openai.OpenAIError as exc:
            raise ClassifierFailed(f"api:{type(exc).__name__}") from None
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
