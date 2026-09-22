"""The document types this service recognises (HospitalAgent design §2). The code is what every
system stores and sends; max_age_days is the validity rule the intake applies."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentType:
    code: str
    label: str
    max_age_days: int


DOCUMENT_TYPES: tuple[DocumentType, ...] = (
    DocumentType("CBC", "ספירת דם מלאה", 90),
    DocumentType("COAGULATION_TESTS", "בדיקות קרישה", 90),
    DocumentType("ECG", "תרשים פעילות חשמלית של הלב", 180),
    DocumentType("URINALYSIS", "בדיקת שתן", 90),
    DocumentType("PREOP_SUMMARY", "סיכום טרום ניתוח", 30),
)
BY_CODE = {t.code: t for t in DOCUMENT_TYPES}
CODES = tuple(t.code for t in DOCUMENT_TYPES)
