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
from .magic import sniff_kind

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

# A scanned PDF (no text layer) sends its first pages' embedded images to vision instead - bounded
# the same way, so a hostile file cannot make this do unbounded work either.
MAX_IMAGES = 4
MAX_IMAGE_DIMENSION = 1600


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
    # A fixed code at every refusal (never set for ACCEPTED or DUPLICATE_DOCUMENT). Logged and
    # returned additively in the 201 body - never document content.
    reason: str | None = None


class _TextExtractionFailed(Exception):
    """A PDF that could not be read. `reason` is one of `parse_error`, `too_many_pages`,
    `too_much_text` - never the document's content."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _text_of(data: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = reader.pages
        if len(pages) > MAX_PAGES:
            raise _TextExtractionFailed("too_many_pages")
        parts: list[str] = []
        total = 0
        for page in pages:
            piece = page.extract_text() or ""
            parts.append(piece)
            total += len(piece)
            if total > MAX_TEXT_BYTES:
                raise _TextExtractionFailed("too_much_text")
        return "".join(parts)
    except _TextExtractionFailed:
        raise
    except Exception:  # any other parse failure is "unreadable", never a crash
        raise _TextExtractionFailed("parse_error") from None


def _downscale(data: bytes) -> bytes | None:
    """Re-encodes one embedded page image as a JPEG no larger than MAX_IMAGE_DIMENSION on its long
    side. `None` if Pillow cannot make sense of it (a corrupt or unsupported embedded image is
    simply skipped, not a crash)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest > MAX_IMAGE_DIMENSION:
                scale = MAX_IMAGE_DIMENSION / longest
                image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=85)
            return out.getvalue()
    except Exception:
        return None


def _page_images(data: bytes) -> list[bytes]:
    """The first pages' embedded images, at most MAX_IMAGES, downscaled - for a PDF whose
    extracted text was empty (a scan). Any failure (a corrupt PDF, an unreadable embedded image)
    yields fewer images rather than raising; an empty result means "no usable image", which the
    caller reports as `no_text_layer`."""
    images: list[bytes] = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception:
        return images
    for page in reader.pages:
        if len(images) >= MAX_IMAGES:
            break
        try:
            page_images = list(page.images)
        except Exception:
            continue
        for image_file in page_images:
            if len(images) >= MAX_IMAGES:
                break
            downscaled = _downscale(image_file.data)
            if downscaled is not None:
                images.append(downscaled)
    return images


def run_intake(data: bytes, patient_id: str, *, classifier: Classifier, today: date,
               is_duplicate: Callable[[str], DuplicateOf | None], max_bytes: int) -> IntakeOutcome:
    sha = hashlib.sha256(data).hexdigest()

    def outcome(result: str, doc_type: str | None = None, doc_date: date | None = None,
               duplicate_of: str | None = None, reason: str | None = None) -> IntakeOutcome:
        return IntakeOutcome(result, doc_type, doc_date, sha, len(data), duplicate_of, reason)

    # 1. size
    if len(data) > max_bytes:
        return outcome(DOCUMENT_UNREADABLE, reason="too_large")
    # 2. signature - %PDF-, JPEG or PNG by magic bytes; anything else is not supported
    kind = sniff_kind(data)
    if kind is None:
        return outcome(DOCUMENT_UNREADABLE, reason="not_supported_format")
    # 3. an image goes straight to vision; a PDF is parsed for its text, and only when that text is
    # empty (a scan) does it fall back to its own page images, also through vision
    text: str | None = None
    images: list[bytes] | None = None
    if kind in ("jpeg", "png"):
        images = [data]
    else:
        try:
            text = _text_of(data)
        except _TextExtractionFailed as exc:
            return outcome(DOCUMENT_UNREADABLE, reason=exc.reason)
        if not text.strip():
            text = None
            images = _page_images(data)
            if not images:
                return outcome(DOCUMENT_UNREADABLE, reason="no_text_layer")
    # 4. an accepted duplicate (a rejected file may be tried again). The original's own type and
    # date travel with the outcome (I4), so a caller whose first request timed out can treat the
    # retry as the document it already delivered.
    original = is_duplicate(sha)
    if original is not None:
        return outcome(DUPLICATE_DOCUMENT, original.document_type, original.document_date, original.document_id)
    # 5. classification. A provider failure (ClassifierUnavailable) is not caught here - it is not
    # a verdict on the file, and propagates to the caller (app/main.py: 503 classifier_unavailable).
    try:
        found = classifier.classify(text) if text is not None else classifier.classify_images(images)
    except ClassifierFailed:
        return outcome(DOCUMENT_UNREADABLE, reason="classifier_unparsable")
    if not found.is_medical:
        return outcome(NON_MEDICAL_DOCUMENT, None, found.document_date)
    doc_type = BY_CODE.get(found.document_type or "")
    if doc_type is None:
        return outcome(DOCUMENT_UNREADABLE, None, found.document_date, reason="unknown_type")
    # 6. the patient's own
    if found.patient_identifier:
        found_ids = set(_IDP_SHAPE.findall(found.patient_identifier.upper()))
        if found_ids and any(found_id != patient_id.upper() for found_id in found_ids):
            return outcome(PATIENT_MISMATCH, doc_type.code, found.document_date)
    # 7. validity (no date is expired; a date after today cannot be right)
    if found.document_date is None:
        return outcome(DOCUMENT_EXPIRED, doc_type.code, None, reason="no_date")
    if found.document_date > today:
        return outcome(DOCUMENT_UNREADABLE, doc_type.code, found.document_date, reason="future_date")
    if (today - found.document_date).days > doc_type.max_age_days:
        return outcome(DOCUMENT_EXPIRED, doc_type.code, found.document_date, reason="too_old")
    return outcome(ACCEPTED, doc_type.code, found.document_date)
