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

def _solid_image(fmt: str) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), color=(10, 20, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def _scanned_pdf() -> bytes:
    """A single-page PDF whose only content is an embedded image - no text layer, exactly what a
    scanner produces - built with Pillow's own PDF writer rather than a hand-crafted fixture."""
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), color=(5, 5, 5)).save(buffer, format="PDF")
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
