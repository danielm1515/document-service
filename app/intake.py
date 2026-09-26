"""The intake, in the design's order (HospitalAgent design §4.2) - the first failure decides."""
from __future__ import annotations

import hashlib
import io
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pypdf
from PIL import Image, ImageOps

from .catalog import BY_CODE
from .classifier import Classifier, ClassifierFailed
from .magic import sniff_kind

# A firm ceiling on decoded pixel count (review I3): Pillow's own default (~89M) only warns below
# 2x itself, so this module additionally turns that warning into an error inside _downscale. Set
# once, at import time - a global on PIL.Image, not per-call state.
Image.MAX_IMAGE_PIXELS = 25_000_000

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
# A page image below this on both sides is an icon or a logo, not a scan (review M1).
MIN_IMAGE_DIMENSION = 200
# Declared (not yet decoded) pixel count above which an embedded image is skipped outright
# (review I3) - kept equal to Image.MAX_IMAGE_PIXELS above.
MAX_DECLARED_PIXELS = 25_000_000
# Fewer than this many non-whitespace characters is treated as no usable text layer, even if
# extraction technically returned something (a header/footer/watermark on an otherwise scanned
# page) - review M5.
MIN_TEXT_CHARS = 40


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


class _ImageTooLarge(Exception):
    """Raised by _downscale whenever a refusal is specifically about pixel count - the file's own
    declared size (a PNG, or a JPEG that fails Pillow's decompression-bomb check outright), or,
    for a standalone JPEG photo (`standalone_jpeg_photo=True`, review N1), the *drafted* size.
    Distinguished from an ordinary decode failure (a corrupt or unsupported image) so a caller
    that reports a reason to the patient can say `too_large` instead of the generic
    `parse_error`. `_page_images` (an embedded PDF page image) still treats this exactly like any
    other failure - that page simply contributes nothing - since the whole-document reason there
    stays `no_text_layer`, never a per-page code."""


def _finish_downscale(image: Image.Image) -> bytes:
    """The tail shared by both _downscale paths: EXIF orientation applied then stripped (the
    final save never re-attaches it - review I1), converted to RGB, resized to at most
    MAX_IMAGE_DIMENSION on the long side, re-encoded as JPEG."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / longest
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _downscale(data: bytes, *, standalone_jpeg_photo: bool = False) -> bytes | None:
    """Re-encodes one image - a standalone upload or an embedded PDF page image - as a JPEG no
    larger than MAX_IMAGE_DIMENSION on its long side (review I1). `None` for anything Pillow
    cannot make sense of; `_ImageTooLarge` specifically for a pixel-count refusal (review N1) -
    for a PNG or an embedded PDF image, that is a decompression-bomb risk (review I3):
    `Image.open` reports pixel dimensions before any pixel is decoded, so an over-declared image
    is rejected there, and a Pillow DecompressionBombWarning (a size Pillow would otherwise only
    warn about) is escalated to an exception for this call - both that escalated warning and
    Pillow's own harder DecompressionBombError (over twice Image.MAX_IMAGE_PIXELS) are converted
    to `_ImageTooLarge`, since either way the refusal is about size, not corruption.

    `standalone_jpeg_photo=True` (review N1) is the one exception, for a standalone JPEG upload
    only: a modern phone photo is legitimately 48-50 megapixels, well over MAX_DECLARED_PIXELS,
    but a JPEG's own `draft()` decodes directly at a reduced scale (libjpeg's own IDCT scaling -
    a real memory bound, not a resize after a full decode), so here the pixel bound is checked
    against the *drafted* size instead of the file's declared size, and the warn-only band is
    deliberately not escalated - only Pillow's hard error above twice MAX_IMAGE_PIXELS, or the
    post-draft bound, can still refuse it, and both still raise `_ImageTooLarge`."""
    try:
        if standalone_jpeg_photo:
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.draft("RGB", (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
                    width, height = image.size
                    if width * height > MAX_DECLARED_PIXELS:
                        raise _ImageTooLarge()
                    return _finish_downscale(image)
            except Image.DecompressionBombError:
                raise _ImageTooLarge() from None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size  # header only - no pixel has been decoded yet
                    if width * height > MAX_DECLARED_PIXELS:
                        raise _ImageTooLarge()
                    if image.format == "JPEG":
                        # Lets libjpeg decode directly at roughly the target size instead of
                        # full resolution - a no-op for any other format (Image.draft's base).
                        image.draft("RGB", (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
                    return _finish_downscale(image)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            # Pillow's own MAX_IMAGE_PIXELS-driven check already fired inside Image.open() -
            # before the explicit check above was ever reached - for anything past its own
            # threshold; either way this is a size refusal, not a decode failure.
            raise _ImageTooLarge() from None
    except _ImageTooLarge:
        raise
    except Exception:
        return None


def _declared_size(page: pypdf.PageObject, path: str | list) -> tuple[int, int] | None:
    """The declared (undecoded) pixel dimensions of one entry from `page.images.keys()` - a bare
    resource name for a top-level image, or a list of names walking down through one or more Form
    XObjects to the actual image dictionary (review M-A: `page.images.keys()` already returns
    such nested paths for a scan wrapped in a Form XObject, e.g. `['/Fm0', '/Im0']`). Reads only
    dictionaries - `/Subtype`, `/Width`, `/Height`, and each Form's own `/Resources` to descend
    into - never a pixel."""
    try:
        names = [path] if isinstance(path, str) else list(path)
        resources = page.get("/Resources")
        xobj = None
        for name in names:
            xobjects = resources.get("/XObject") if resources else None
            if not xobjects or name not in xobjects:
                return None
            xobj = xobjects[name]
            if xobj.get("/Subtype") == "/Form":
                resources = xobj.get("/Resources")
        if xobj is None or xobj.get("/Subtype") != "/Image":
            return None
        return int(xobj.get("/Width", 0)), int(xobj.get("/Height", 0))
    except Exception:
        return None


def _largest_usable_image_name(page: pypdf.PageObject):
    """The `page.images` key of the largest embedded image on this page worth sending to vision -
    not a tiny icon or logo (review M1), not so large its *declared* pixel count alone is a
    decompression-bomb risk (review I3), and found even when it is nested inside a Form XObject
    (review M-A). `page.images.keys()` already walks Form XObjects to find every image path;
    `_declared_size` reads each one's dictionary without decoding. `None` if the page has no
    usable image."""
    try:
        paths = list(page.images.keys())
    except Exception:
        return None
    best_path, best_area = None, 0
    for path in paths:
        size = _declared_size(page, path)
        if size is None:
            continue
        width, height = size
        if width < MIN_IMAGE_DIMENSION and height < MIN_IMAGE_DIMENSION:
            continue  # an icon or a logo, not a page scan
        area = width * height
        if area == 0 or area > MAX_DECLARED_PIXELS:
            continue
        if area > best_area:
            best_path, best_area = path, area
    return best_path


def _page_images(data: bytes) -> list[bytes]:
    """The largest usable embedded image on each of the first pages, at most MAX_IMAGES - for a
    PDF whose extracted text was empty or too short (a scan). A page whose only images are tiny,
    oversized or fail to decode contributes nothing - it is not retried with a smaller candidate
    on the same page (review M1's "fail closed for that page"); an empty result means "no usable
    image anywhere", which the caller reports as `no_text_layer`."""
    images: list[bytes] = []
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception:
        return images
    for page in reader.pages:
        if len(images) >= MAX_IMAGES:
            break
        path = _largest_usable_image_name(page)
        if path is None:
            continue
        try:
            raw = page.images[path].data
        except Exception:
            continue
        try:
            downscaled = _downscale(raw)
        except _ImageTooLarge:
            continue  # this page contributes nothing - no per-page reason surfaces (M1)
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
    # 3. an image goes straight to vision (downscaled/EXIF-stripped like any page image - review
    # I1); a PDF is parsed for its text, and only when that text is too short to be useful (review
    # M5) does it fall back to its own page images, also through vision
    text: str | None = None
    images: list[bytes] | None = None
    if kind in ("jpeg", "png"):
        try:
            # A standalone JPEG gets the draft-scaled path (review N1): a 48-50MP phone photo is
            # legitimate and must not be refused just for its declared size. PNG has no draft
            # scaling, so it stays on the ordinary path, unaffected by this review round.
            downscaled = _downscale(data, standalone_jpeg_photo=(kind == "jpeg"))
        except _ImageTooLarge:
            return outcome(DOCUMENT_UNREADABLE, reason="too_large")
        if downscaled is None:
            return outcome(DOCUMENT_UNREADABLE, reason="parse_error")
        images = [downscaled]
    else:
        try:
            text = _text_of(data)
        except _TextExtractionFailed as exc:
            return outcome(DOCUMENT_UNREADABLE, reason=exc.reason)
        if len(re.sub(r"\s+", "", text)) < MIN_TEXT_CHARS:
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
