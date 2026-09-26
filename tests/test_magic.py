from app.magic import mime_of, sniff_kind


def test_sniff_kind_recognises_pdf_jpeg_and_png_by_magic_bytes():
    assert sniff_kind(b"%PDF-1.7 rest of the file") == "pdf"
    assert sniff_kind(b"\xff\xd8\xffrest of a jpeg") == "jpeg"
    assert sniff_kind(b"\x89PNG\r\n\x1a\nrest of a png") == "png"


def test_sniff_kind_is_none_for_anything_else():
    assert sniff_kind(b"") is None
    assert sniff_kind(b"hello") is None
    assert sniff_kind(b"GIF89a") is None


def test_mime_of_matches_the_sniffed_kind_and_defaults_to_jpeg():
    assert mime_of(b"\xff\xd8\xffrest") == "image/jpeg"
    assert mime_of(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert mime_of(b"not an image at all") == "image/jpeg"
