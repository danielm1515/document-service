from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One upload, accepted or not. Only an accepted one has an s3_object_key.

    `seq` is the primary key - a plain autoincrementing integer, which SQLite (and every other
    backend) grants for free to a sole-integer primary key - kept only to break a tie in
    `uploaded_at` with the true insertion order (minor: uploaded_at then an insertion sequence).
    `document_id` (the id every API response and every other table uses) stays a unique, indexed
    column rather than the primary key."""

    __tablename__ = "documents"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    document_type: Mapped[str | None] = mapped_column(String(32))
    document_date: Mapped[date | None] = mapped_column(Date)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    s3_object_key: Mapped[str | None] = mapped_column(String(200))
    # Set in Python (app/main.py's _utc_now), never a DB default: a DB-side default would lose the
    # microseconds and the UTC offset that the API response must emit (minor).
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                   default=lambda: datetime.now(timezone.utc), nullable=False)


class AuditLog(Base):
    """Codes and ids only - never a file name, text or content."""

    __tablename__ = "document_audit_logs"

    audit_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    document_id: Mapped[str | None] = mapped_column(String(32))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
