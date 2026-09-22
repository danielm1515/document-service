from datetime import date
from pathlib import Path

import pytest

from app.classifier import Classification, ClassifierFailed, FakeClassifier
from app.intake import (ACCEPTED, DOCUMENT_EXPIRED, DOCUMENT_UNREADABLE, DUPLICATE_DOCUMENT, NON_MEDICAL_DOCUMENT,
                        PATIENT_MISMATCH, run_intake)

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 9, 22)
MAX = 10 * 1024 * 1024


def data(year, name):
    return (FIXTURES / year / f"{name}.pdf").read_bytes()


def intake(raw, *, classifier=None, today=TODAY, duplicates=(), patient="P-10041"):
    return run_intake(raw, patient, classifier=classifier or FakeClassifier(), today=today,
                      is_duplicate=lambda sha: sha in duplicates, max_bytes=MAX)


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

    class Boom:
        def classify(self, text):
            raise AssertionError("a duplicate must not reach the classifier")
    assert intake(raw, classifier=Boom(), duplicates={first.sha256}).result == DUPLICATE_DOCUMENT


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
