from datetime import date
from pathlib import Path

import pytest

from app.classifier import Classification, ClassifierFailed, FakeClassifier
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


@pytest.mark.parametrize("raw", [b"", b"hello", b"%PDF-1.7 but not really a pdf", b"GIF89a"])
def test_something_that_is_not_a_readable_pdf_is_unreadable(raw):
    assert intake(raw).result == DOCUMENT_UNREADABLE


def test_a_file_over_the_size_limit_is_unreadable_without_being_parsed():
    class Boom:
        def classify(self, text):
            raise AssertionError("must not be called")
    assert intake(b"%PDF-" + b"0" * MAX, classifier=Boom()).result == DOCUMENT_UNREADABLE


def test_a_pdf_with_no_text_is_unreadable():
    import io
    import pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    assert intake(buffer.getvalue()).result == DOCUMENT_UNREADABLE


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


@pytest.mark.parametrize("answer, result", [
    (ClassifierFailed("x"), DOCUMENT_UNREADABLE),                                   # no confident answer
    (Classification(True, None, date(2026, 9, 1), None), DOCUMENT_UNREADABLE),      # medical, no catalog type
    (Classification(True, "CBC", None, None), DOCUMENT_EXPIRED),                    # no date: fail closed
    (Classification(True, "CBC", date(2026, 9, 23), None), DOCUMENT_UNREADABLE),    # dated after today
    (Classification(True, "CBC", date(2026, 9, 1), "P-20000"), PATIENT_MISMATCH),   # another patient's
    (Classification(True, "CBC", date(2026, 9, 1), "P-10041"), ACCEPTED),           # the patient's own
    (Classification(True, "CBC", date(2026, 9, 1), "402781"), ACCEPTED),            # not the IdP shape: ignored
    (Classification(False, "CBC", date(2026, 9, 1), None), NON_MEDICAL_DOCUMENT),   # not medical wins
])
def test_each_classification_outcome(answer, result):
    assert intake(data("2026", "cbc"), classifier=Scripted(answer)).result == result


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


@pytest.mark.parametrize("doc_type, age_days, result", [
    ("CBC", 90, ACCEPTED), ("CBC", 91, DOCUMENT_EXPIRED),
    ("ECG", 180, ACCEPTED), ("ECG", 181, DOCUMENT_EXPIRED),
    ("PREOP_SUMMARY", 30, ACCEPTED), ("PREOP_SUMMARY", 31, DOCUMENT_EXPIRED),
])
def test_validity_is_per_type_and_inclusive(doc_type, age_days, result):
    from datetime import timedelta
    answer = Classification(True, doc_type, TODAY - timedelta(days=age_days), None)
    assert intake(data("2026", "cbc"), classifier=Scripted(answer)).result == result


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
    assert intake(_merged_pdf(21)).result == DOCUMENT_UNREADABLE


def test_a_pdf_with_20_pages_is_accepted():
    assert intake(_merged_pdf(20)).result == ACCEPTED
