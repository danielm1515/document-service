"""The fixtures the intake tests use: both sets present, readable, and dated as expected."""
import re
from pathlib import Path

import pypdf
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = ("cbc", "coagulation", "ecg", "urinalysis", "preop_summary", "electricity_bill")


def text_of(path: Path) -> str:
    return "".join(page.extract_text() or "" for page in pypdf.PdfReader(path).pages)


@pytest.mark.parametrize("year", ["2024", "2026"])
@pytest.mark.parametrize("name", NAMES)
def test_every_fixture_is_a_readable_pdf_with_its_year(year, name):
    path = FIXTURES / year / f"{name}.pdf"
    assert path.read_bytes().startswith(b"%PDF-")
    text = text_of(path)
    assert text.strip()
    first_date = re.search(r"\d{2}\.\d{2}\.(\d{4})", text)
    assert first_date and first_date.group(1) == year


def test_the_2026_set_is_issued_on_15_september():
    assert "15.09.2026" in text_of(FIXTURES / "2026" / "cbc.pdf")
