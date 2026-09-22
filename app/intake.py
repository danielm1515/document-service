"""The intake, in the design's order (HospitalAgent design §4.2) - the first failure decides."""
from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pypdf

from .catalog import BY_CODE
from .classifier import Classifier, ClassifierFailed

ACCEPTED = "ACCEPTED"
NON_MEDICAL_DOCUMENT = "NON_MEDICAL_DOCUMENT"
DOCUMENT_UNREADABLE = "DOCUMENT_UNREADABLE"
DOCUMENT_EXPIRED = "DOCUMENT_EXPIRED"
DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
PATIENT_MISMATCH = "PATIENT_MISMATCH"

# Only an identifier of the demo IdP's shape is compared; any other number the model reports
# (a document or customer number) is ignored rather than guessed at (design §4.2). Matched
# case-insensitively and anywhere in the field - the model may return surrounding text
# ("מטופל: P-20000") rather than the bare id.
_IDP_SHAPE = re.compile(r"P-\d+")

# A crafted PDF within the 10 MB size limit can still decompress to many pages or a lot of text;
# the demo documents are one page. Both bounds make DOCUMENT_UNREADABLE instead of doing
# unbounded work on hostile input.
MAX_PAGES = 20
MAX_TEXT_BYTES = 200_000


@dataclass(frozen=True)
class DuplicateOf:
    """The accepted original a duplicate upload matches (design §4.1, I4). A small value object,
    not a SQLAlchemy row, so this module stays free of SQLAlchemy - `app/main.py` builds it from
    the `Document` it looked up."""
    document_id: str
    document_type: str | None
    document_date: date | None


@dataclass(frozen=True)
class IntakeOutcome:
    result: str
    document_type: str | None
    document_date: date | None
    sha256: str
    size_bytes: int
    duplicate_of: str | None = None


def _text_of(data: bytes) -> str | None:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if len(reader.pages) > MAX_PAGES:
            return None
        parts: list[str] = []
        total = 0
        for page in reader.pages:
            piece = page.extract_text() or ""
            parts.append(piece)
            total += len(piece)
            if total > MAX_TEXT_BYTES:
                return None
        return "".join(parts)
    except Exception:  # any parse failure is "unreadable", never a crash
        return None


def run_intake(data: bytes, patient_id: str, *, classifier: Classifier, today: date,
               is_duplicate: Callable[[str], DuplicateOf | None], max_bytes: int) -> IntakeOutcome:
    sha = hashlib.sha256(data).hexdigest()

    def outcome(result: str, doc_type: str | None = None, doc_date: date | None = None,
               duplicate_of: str | None = None) -> IntakeOutcome:
        return IntakeOutcome(result, doc_type, doc_date, sha, len(data), duplicate_of)

    # 1. size and signature
    if len(data) > max_bytes or not data.startswith(b"%PDF-"):
        return outcome(DOCUMENT_UNREADABLE)
    # 2. parse and extract text
    text = _text_of(data)
    if not text or not text.strip():
        return outcome(DOCUMENT_UNREADABLE)
    # 3. an accepted duplicate (a rejected file may be tried again). The original's own type and
    # date travel with the outcome (I4), so a caller whose first request timed out can treat the
    # retry as the document it already delivered.
    original = is_duplicate(sha)
    if original is not None:
        return outcome(DUPLICATE_DOCUMENT, original.document_type, original.document_date, original.document_id)
    # 4. classification
    try:
        found = classifier.classify(text)
    except ClassifierFailed:
        return outcome(DOCUMENT_UNREADABLE)
    if not found.is_medical:
        return outcome(NON_MEDICAL_DOCUMENT, None, found.document_date)
    doc_type = BY_CODE.get(found.document_type or "")
    if doc_type is None:
        return outcome(DOCUMENT_UNREADABLE, None, found.document_date)
    # 5. the patient's own
    if found.patient_identifier:
        found_ids = set(_IDP_SHAPE.findall(found.patient_identifier.upper()))
        if found_ids and any(found_id != patient_id.upper() for found_id in found_ids):
            return outcome(PATIENT_MISMATCH, doc_type.code, found.document_date)
    # 6. validity (no date is expired; a date after today cannot be right)
    if found.document_date is None:
        return outcome(DOCUMENT_EXPIRED, doc_type.code, None)
    if found.document_date > today:
        return outcome(DOCUMENT_UNREADABLE, doc_type.code, found.document_date)
    if (today - found.document_date).days > doc_type.max_age_days:
        return outcome(DOCUMENT_EXPIRED, doc_type.code, found.document_date)
    return outcome(ACCEPTED, doc_type.code, found.document_date)
