"""Magic-byte sniffing shared by the intake (what kind of file was uploaded) and the classifier
(what MIME type to declare for an image sent to vision) - kept in its own module so neither one
has to import the other for it."""
from __future__ import annotations

PDF = b"%PDF-"
JPEG = b"\xff\xd8\xff"
PNG = b"\x89PNG\r\n\x1a\n"

EXTENSION = {"pdf": "pdf", "jpeg": "jpg", "png": "png"}
CONTENT_TYPE = {"pdf": "application/pdf", "jpeg": "image/jpeg", "png": "image/png"}


def sniff_kind(data: bytes) -> str | None:
    """`None` for anything that is not one of the three accepted formats (design §4.2)."""
    if data.startswith(PDF):
        return "pdf"
    if data.startswith(JPEG):
        return "jpeg"
    if data.startswith(PNG):
        return "png"
    return None


def mime_of(image: bytes) -> str:
    """The MIME type of a standalone image or a page image pulled out of a PDF - defaults to
    JPEG (what `_downscale` always re-encodes a PDF's embedded images as)."""
    kind = sniff_kind(image)
    return CONTENT_TYPE.get(kind, "image/jpeg")
