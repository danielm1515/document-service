import io
from datetime import date

from pathlib import Path

import pytest

from app.classifier import Classification, ClassifierFailed, ClassifierUnavailable, FakeClassifier
from app.intake import (ACCEPTED, DOCUMENT_EXPIRED, DOCUMENT_UNREADABLE, DUPLICATE_DOCUMENT, NON_MEDICAL_DOCUMENT,
                        PATIENT_MISMATCH, DuplicateOf, run_intake)

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 9, 22)
MAX = 10 * 1024 * 1024


def data(year, name):
    return (FIXTURES / year / f"{name}.pdf").read_bytes()


def intake(raw, *, classifier=None, today=TODAY, duplicates=None, patient="P-10041"):
    duplicates = duplicates or {}
    return run_intake(raw, patient, classifier=classifier or FakeClassifier(), today=today,
                      is_duplicate=lambda sha: duplicates.get(sha), max_bytes=MAX)


@pytest.mark.parametrize("name, doc_type", [("cbc", "CBC"), ("coagulation", "COAGULATION_TESTS"), ("ecg", "ECG"),
                                            ("urinalysis", "URINALYSIS"), ("preop_summary", "PREOP_SUMMARY")])
def test_the_2026_medical_files_are_accepted_with_their_type(name, doc_type):
    outcome = intake(data("2026", name))
    assert (outcome.result, outcome.document_type, outcome.document_date) == (ACCEPTED, doc_type, date(2026, 9, 15))
    assert len(outcome.sha256) == 64 and outcome.size_bytes == len(data("2026", name))


@pytest.mark.parametrize("name", ["cbc", "coagulation", "ecg", "urinalysis", "preop_summary"])
def test_the_2024_originals_are_expired(name):
    assert intake(data("2024", name)).result == DOCUMENT_EXPIRED


@pytest.mark.parametrize("year", ["2024", "2026"])
def test_the_electricity_bill_is_not_medical(year):
    assert intake(data(year, "electricity_bill")).result == NON_MEDICAL_DOCUMENT


@pytest.mark.parametrize("raw", [b"", b"hello", b"GIF89a"])
def test_something_of_an_unsupported_format_is_unreadable(raw):
    """Not %PDF-, JPEG or PNG by magic bytes (Task 2, decision 3)."""
    outcome = intake(raw)
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "not_supported_format"


def test_a_pdf_that_fails_to_parse_has_the_parse_error_reason():
    """Starts with the PDF magic bytes, but pypdf cannot make sense of the rest."""
    outcome = intake(b"%PDF-1.7 but not really a pdf")
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "parse_error"


def test_a_file_over_the_size_limit_is_unreadable_without_being_parsed():
    class Boom:
        def classify(self, text):
            raise AssertionError("must not be called")
    outcome = intake(b"%PDF-" + b"0" * MAX, classifier=Boom())
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "too_large"


def test_a_pdf_with_no_text_and_no_embedded_image_has_the_no_text_layer_reason():
    import pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    outcome = intake(buffer.getvalue())
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "no_text_layer"


def test_too_much_extracted_text_has_the_too_much_text_reason(monkeypatch):
    """Bounded the same way as too_many_pages, but a per-page string is cheap to fake, so this
    exercises app.intake._text_of directly rather than needing a real giant-text fixture."""
    import app.intake as intake_module

    class FakePage:
        def extract_text(self):
            return "x" * 90_000

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [FakePage(), FakePage(), FakePage()]

    monkeypatch.setattr(intake_module.pypdf, "PdfReader", FakeReader)
    outcome = intake(b"%PDF-fake")
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "too_much_text"


def test_an_accepted_duplicate_is_refused_before_classification():
    raw = data("2026", "cbc")
    first = intake(raw)
    original = DuplicateOf("DOC-ORIGINAL1", "CBC", date(2026, 9, 15))

    class Boom:
        def classify(self, text):
            raise AssertionError("a duplicate must not reach the classifier")
    outcome = intake(raw, classifier=Boom(), duplicates={first.sha256: original})
    assert outcome.result == DUPLICATE_DOCUMENT


def test_a_duplicate_outcome_carries_the_original_document():
    raw = data("2026", "cbc")
    sha = intake(raw).sha256
    original = DuplicateOf("DOC-ORIGINAL1", "CBC", date(2026, 9, 15))
    outcome = intake(raw, duplicates={sha: original})
    assert outcome.result == DUPLICATE_DOCUMENT
    assert outcome.duplicate_of == "DOC-ORIGINAL1"
    assert outcome.document_type == "CBC" and outcome.document_date == date(2026, 9, 15)


def test_a_non_duplicate_outcome_has_no_duplicate_of():
    assert intake(data("2026", "cbc")).duplicate_of is None


class Scripted:
    def __init__(self, answer):
        self.answer = answer

    def classify(self, text):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.parametrize("answer, result, reason", [
    (ClassifierFailed("x"), DOCUMENT_UNREADABLE, "classifier_unparsable"),                 # no confident answer
    (Classification(True, None, date(2026, 9, 1), None), DOCUMENT_UNREADABLE, "unknown_type"),  # medical, no catalog type
    (Classification(True, "CBC", None, None), DOCUMENT_EXPIRED, "no_date"),                # no date: fail closed
    (Classification(True, "CBC", date(2026, 9, 23), None), DOCUMENT_UNREADABLE, "future_date"),  # dated after today
    (Classification(True, "CBC", date(2026, 9, 1), "P-20000"), PATIENT_MISMATCH, None),    # another patient's
    (Classification(True, "CBC", date(2026, 9, 1), "P-10041"), ACCEPTED, None),             # the patient's own
    (Classification(True, "CBC", date(2026, 9, 1), "402781"), ACCEPTED, None),              # not the IdP shape: ignored
    (Classification(False, "CBC", date(2026, 9, 1), None), NON_MEDICAL_DOCUMENT, None),    # not medical wins
])
def test_each_classification_outcome(answer, result, reason):
    outcome = intake(data("2026", "cbc"), classifier=Scripted(answer))
    assert outcome.result == result and outcome.reason == reason


def test_a_classifier_unavailable_error_is_not_caught_by_run_intake():
    """A provider failure is never a verdict on the file (Task 2, decision 1) - it propagates to
    the caller (app/main.py), which turns it into 503 classifier_unavailable."""
    class Unavailable:
        def classify(self, text):
            raise ClassifierUnavailable("api:RateLimitError")
    with pytest.raises(ClassifierUnavailable):
        intake(data("2026", "cbc"), classifier=Unavailable())


# --- Minor: the identifier check is re.findall(r"P-\d+", identifier.upper()) - case-insensitive,
# and matched anywhere in the string, not only when the whole field is the identifier. ---

@pytest.mark.parametrize("identifier", ["p-20000", "מטופל: P-20000"])
def test_patient_mismatch_is_found_case_insensitively_and_inside_surrounding_text(identifier):
    answer = Classification(True, "CBC", date(2026, 9, 1), identifier)
    assert intake(data("2026", "cbc"), classifier=Scripted(answer)).result == PATIENT_MISMATCH


@pytest.mark.parametrize("identifier", ["p-10041", "מטופל: P-10041"])
def test_the_patients_own_id_is_still_accepted_case_insensitively_and_inside_surrounding_text(identifier):
    answer = Classification(True, "CBC", date(2026, 9, 1), identifier)
    assert intake(data("2026", "cbc"), classifier=Scripted(answer)).result == ACCEPTED


@pytest.mark.parametrize("doc_type, age_days, result, reason", [
    ("CBC", 90, ACCEPTED, None), ("CBC", 91, DOCUMENT_EXPIRED, "too_old"),
    ("ECG", 180, ACCEPTED, None), ("ECG", 181, DOCUMENT_EXPIRED, "too_old"),
    ("PREOP_SUMMARY", 30, ACCEPTED, None), ("PREOP_SUMMARY", 31, DOCUMENT_EXPIRED, "too_old"),
])
def test_validity_is_per_type_and_inclusive(doc_type, age_days, result, reason):
    from datetime import timedelta
    answer = Classification(True, doc_type, TODAY - timedelta(days=age_days), None)
    outcome = intake(data("2026", "cbc"), classifier=Scripted(answer))
    assert outcome.result == result and outcome.reason == reason


def test_a_rejected_outcome_keeps_its_hash_and_size():
    outcome = intake(data("2026", "electricity_bill"))
    assert outcome.document_type is None and len(outcome.sha256) == 64 and outcome.size_bytes > 0


def _merged_pdf(pages):
    import io
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data("2026", "cbc")))
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_page(reader.pages[0])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_a_pdf_with_more_than_20_pages_is_unreadable():
    outcome = intake(_merged_pdf(21))
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "too_many_pages"


def test_a_pdf_with_20_pages_is_accepted():
    assert intake(_merged_pdf(20)).result == ACCEPTED


# --- Task 2, decision 3: an image, or a scanned PDF with no text layer, goes to
# classifier.classify_images instead - same duplicate/type/ownership/validity logic downstream.
# FakeClassifier.classify_images always answers CBC dated 2026-09-15 (see app/classifier.py). ---

def _solid_image(fmt: str, size: tuple[int, int] = (40, 40)) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(10, 20, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def _scanned_pdf(size: tuple[int, int] = (800, 600)) -> bytes:
    """A single-page PDF whose only content is an embedded image - no text layer, exactly what a
    scanner produces - built with Pillow's own PDF writer rather than a hand-crafted fixture."""
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(5, 5, 5)).save(buffer, format="PDF")
    return buffer.getvalue()


def _multi_page_scan(count: int, size: tuple[int, int] = (600, 400)) -> bytes:
    """`count` pages, each with its own embedded image - built by merging `count` one-image PDFs
    (Pillow can only ever write a single-image page itself)."""
    import pypdf
    writer = pypdf.PdfWriter()
    for i in range(count):
        reader = pypdf.PdfReader(io.BytesIO(_scanned_pdf(size)))
        writer.add_page(reader.pages[0])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_a_jpeg_image_is_classified_through_vision():
    outcome = intake(_solid_image("JPEG"))
    assert outcome.result == ACCEPTED and outcome.document_type == "CBC" and outcome.reason is None


def test_a_png_image_is_classified_through_vision():
    outcome = intake(_solid_image("PNG"))
    assert outcome.result == ACCEPTED and outcome.document_type == "CBC" and outcome.reason is None


def test_a_scanned_pdf_with_an_embedded_image_is_classified_through_vision():
    outcome = intake(_scanned_pdf())
    assert outcome.result == ACCEPTED and outcome.document_type == "CBC" and outcome.reason is None


def test_an_image_upload_is_still_subject_to_the_duplicate_check():
    raw = _solid_image("JPEG")
    first = intake(raw)
    original = DuplicateOf("DOC-ORIGINAL1", "CBC", date(2026, 9, 15))

    class Boom:
        def classify_images(self, images):
            raise AssertionError("a duplicate must not reach the classifier")
    outcome = intake(raw, classifier=Boom(), duplicates={first.sha256: original})
    assert outcome.result == DUPLICATE_DOCUMENT


# --- Review round 1, I1: a standalone image is downscaled/EXIF-stripped exactly like an
# embedded page image, not sent raw. A decode failure is DOCUMENT_UNREADABLE/parse_error, never
# a 503 (that stays reserved for a provider failure). ---

def _truncated_jpeg() -> bytes:
    full = _solid_image("JPEG", (300, 200))
    return full[: len(full) // 2]


def _jpeg_with_exif(size=(300, 100)) -> bytes:
    """A JPEG whose EXIF says "rotate 90 degrees" and carries an arbitrary comment tag - both
    should be gone from what actually reaches the classifier."""
    from PIL import Image
    image = Image.new("RGB", size, color=(40, 60, 80))
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    exif[0x9286] = "a free-text EXIF comment, never sent to the model"  # UserComment
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    return buffer.getvalue()


class Spy:
    """Captures whatever run_intake actually sends to vision, without answering anything itself
    but a fixed CBC (like FakeClassifier) - for asserting on the images argument."""

    def __init__(self):
        self.images: list[bytes] | None = None

    def classify(self, text):
        raise AssertionError("the vision path must not fall back to text")

    def classify_images(self, images):
        self.images = list(images)
        return Classification(True, "CBC", date(2026, 9, 15), None)


def test_a_truncated_jpeg_upload_has_the_parse_error_reason():
    outcome = intake(_truncated_jpeg())
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "parse_error"


def test_a_jpeg_uploads_exif_is_stripped_and_its_orientation_applied():
    from PIL import Image
    spy = Spy()
    outcome = intake(_jpeg_with_exif(), classifier=spy)
    assert outcome.result == ACCEPTED
    [sent] = spy.images
    assert b"Exif" not in sent
    reopened = Image.open(io.BytesIO(sent))
    assert dict(reopened.getexif()) == {}
    assert reopened.size == (100, 300)  # the 90-degree rotation swapped width and height


# --- Review round 1, I3: a decompression-bomb-sized image is skipped (or refused), never fully
# decoded. A crafted file can declare a huge size in its header while staying tiny on disk. ---

def _png_declaring_huge_dimensions(width: int = 20000, height: int = 20000) -> bytes:
    """A syntactically valid but tiny PNG (a real IHDR chunk, a throwaway IDAT never meant to
    actually decode) that declares `width`x`height` pixels - Pillow reads that from the header
    alone, without ever touching IDAT, so this is instant regardless of the declared size."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = chunk(b"IDAT", zlib.compress(b"\x00" * 3))  # never actually decoded
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + idat + chunk(b"IEND", b"")


def _pdf_with_an_oversized_declared_image() -> bytes:
    """A page whose one embedded image XObject declares 20000x20000 pixels in its dictionary,
    with a 10-byte garbage stream that is never actually decoded - _largest_usable_image_name
    must reject it from the dictionary alone."""
    import pypdf
    from pypdf.generic import DictionaryObject, NameObject, NumberObject, StreamObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    xobj = StreamObject()
    xobj.set_data(b"\x00" * 10)
    xobj[NameObject("/Type")] = NameObject("/XObject")
    xobj[NameObject("/Subtype")] = NameObject("/Image")
    xobj[NameObject("/Width")] = NumberObject(20000)
    xobj[NameObject("/Height")] = NumberObject(20000)
    xobj[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
    xobj[NameObject("/BitsPerComponent")] = NumberObject(8)
    ref = writer._add_object(xobj)
    if "/Resources" not in page:
        page[NameObject("/Resources")] = DictionaryObject()
    resources = page["/Resources"]
    if "/XObject" not in resources:
        resources[NameObject("/XObject")] = DictionaryObject()
    resources["/XObject"][NameObject("/Im0")] = ref
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_a_standalone_image_declaring_a_huge_size_is_refused_quickly_as_parse_error():
    outcome = intake(_png_declaring_huge_dimensions())
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "parse_error"


def test_a_standalone_image_in_the_warn_only_band_is_still_refused():
    """36,000,000 declared pixels - above MAX_DECLARED_PIXELS (25M) but below Pillow's own
    default hard-error threshold (2x its MAX_IMAGE_PIXELS), so Pillow would only warn unless the
    warning is escalated to an error inside _downscale."""
    outcome = intake(_png_declaring_huge_dimensions(6000, 6000))
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "parse_error"


def test_a_pdf_page_with_only_an_oversized_declared_image_yields_no_text_layer():
    """The garbage 10-byte stream would fail to decode as a 20000x20000 image anyway - this
    proves the rejection happens from the declared size, since app.intake._largest_usable_image_name
    never calls page.images (which would attempt exactly that failing decode)."""
    from app.intake import _largest_usable_image_name
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(_pdf_with_an_oversized_declared_image()))
    assert _largest_usable_image_name(reader.pages[0]) is None
    outcome = intake(_pdf_with_an_oversized_declared_image())
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "no_text_layer"


# --- Review round 1, I4: the vision path carries every downstream check (validity, ownership,
# type) exactly like the text path, is bounded to MAX_IMAGES per document, downscales a large
# embedded scan, and accepts whatever colour mode Pillow hands back for an embedded image. ---

class ScriptedImages:
    """Like test_intake.Scripted, but for the vision path - classify_images answers a fixed
    Classification (or raises), and classify() must never be called."""

    def __init__(self, answer):
        self.answer = answer

    def classify(self, text):
        raise AssertionError("the vision path must not fall back to text")

    def classify_images(self, images):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.parametrize("answer, result, reason", [
    (Classification(True, "CBC", date(2020, 1, 1), None), DOCUMENT_EXPIRED, "too_old"),
    (Classification(True, "CBC", date(2026, 9, 1), "P-20000"), PATIENT_MISMATCH, None),
    (Classification(True, None, date(2026, 9, 1), None), DOCUMENT_UNREADABLE, "unknown_type"),
])
def test_each_classification_outcome_through_vision(answer, result, reason):
    outcome = intake(_solid_image("JPEG"), classifier=ScriptedImages(answer))
    assert outcome.result == result and outcome.reason == reason


def test_a_large_embedded_scan_is_downscaled_to_at_most_1600_on_the_long_side():
    from PIL import Image
    spy = Spy()
    intake(_scanned_pdf((4000, 3000)), classifier=spy)
    [sent] = spy.images
    width, height = Image.open(io.BytesIO(sent)).size
    assert max(width, height) <= 1600


def test_a_six_page_scan_sends_exactly_four_images():
    spy = Spy()
    intake(_multi_page_scan(6), classifier=spy)
    assert len(spy.images) == 4


def test_an_rgba_image_converts_without_crashing():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGBA", (50, 50), color=(1, 2, 3, 128)).save(buffer, format="PNG")
    outcome = intake(buffer.getvalue())
    assert outcome.result == ACCEPTED


def test_a_palette_image_converts_without_crashing():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("P", (50, 50)).save(buffer, format="PNG")
    outcome = intake(buffer.getvalue())
    assert outcome.result == ACCEPTED


# --- Review round 1, M1: image selection skips a tiny image and takes the largest per page. ---

def test_a_tiny_embedded_image_is_skipped_as_an_icon_not_a_scan():
    """Both sides under MIN_IMAGE_DIMENSION (a logo, not a page) - the page contributes nothing,
    and since it is the only page, the whole document is no_text_layer."""
    outcome = intake(_scanned_pdf((100, 80)))
    assert outcome.result == DOCUMENT_UNREADABLE and outcome.reason == "no_text_layer"


def test_the_largest_image_on_a_page_is_the_one_sent():
    import pypdf
    from pypdf.generic import DictionaryObject, NameObject, NumberObject, StreamObject

    from app.intake import _largest_usable_image_name

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    resources = DictionaryObject()
    xobjects = DictionaryObject()
    for name, (w, h) in (("/Small", (300, 300)), ("/Big", (900, 900))):
        xobj = StreamObject()
        xobj.set_data(b"\x00" * 10)
        xobj[NameObject("/Type")] = NameObject("/XObject")
        xobj[NameObject("/Subtype")] = NameObject("/Image")
        xobj[NameObject("/Width")] = NumberObject(w)
        xobj[NameObject("/Height")] = NumberObject(h)
        xobjects[NameObject(name)] = writer._add_object(xobj)
    resources[NameObject("/XObject")] = xobjects
    page[NameObject("/Resources")] = resources
    buffer = io.BytesIO()
    writer.write(buffer)
    reader = pypdf.PdfReader(io.BytesIO(buffer.getvalue()))
    assert _largest_usable_image_name(reader.pages[0]) == "/Big"


# --- Review round 1, M5: too little extracted text (under 40 non-space characters) takes the
# vision path, even though extraction technically returned something - a watermark or a page
# number must not block the vision fallback the way truly empty text already did. ---

def test_a_short_watermark_of_a_text_layer_still_takes_the_vision_path(monkeypatch):
    """Isolates the MIN_TEXT_CHARS threshold: a page reporting real but short text (under 40
    non-space characters) must still reach classify_images, not classify(text) - proven by
    replacing _page_images with a canned answer and checking that is exactly what the classifier
    receives."""
    import app.intake as intake_module

    class FakePage:
        def extract_text(self):
            return "עותק - לא לשימוש"  # 15 non-space characters, well under MIN_TEXT_CHARS

        def get(self, _key):
            return None

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [FakePage()]

    monkeypatch.setattr(intake_module.pypdf, "PdfReader", FakeReader)
    monkeypatch.setattr(intake_module, "_page_images", lambda data: [b"fake-page-image"])
    spy = Spy()
    outcome = intake(b"%PDF-fake", classifier=spy)
    assert outcome.result == ACCEPTED
    assert spy.images == [b"fake-page-image"]
