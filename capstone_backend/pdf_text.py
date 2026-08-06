"""PDF text extraction with OCR as the fallback, not the first resort.

The pipelines originally went straight to ``pdf2image.convert_from_bytes`` plus
pytesseract for every upload. That is the right path for a *scanned* contract, but
it is the wrong default for two reasons:

1. It needs two external binaries - poppler (``pdftoppm``) and tesseract - neither
   of which ships with Python. On a machine without them every upload dies with
   "Unable to get page count. Is poppler installed and in PATH?", which is exactly
   what the dashboard was showing.
2. Most contracts are digital-born and already carry a text layer. Rasterising them
   to 300 DPI images and running OCR over the result is slower and *less accurate*
   than reading the text that is already there.

So: read the embedded text layer first, and fall back to OCR only when the document
genuinely has no text (a scan). If OCR is needed but unavailable, the error says
what to install instead of leaking a poppler stack trace.
"""

import logging
import shutil
from typing import List

logger = logging.getLogger(__name__)

# Below this many characters a "text layer" is almost certainly page furniture
# (a header, a stamp) rather than the contract body, so OCR is worth trying.
MIN_TEXT_LAYER_CHARS = 200

OCR_MISSING_MESSAGE = (
    "This PDF has no text layer, so it needs OCR - but the OCR tools are not "
    "installed. Install poppler (pdftoppm) and tesseract and put both on PATH, or "
    "upload a text-based PDF, or paste the contract text directly."
)


class PdfTextUnavailable(RuntimeError):
    """Raised when neither the text layer nor OCR can produce any text."""


def ocr_available() -> bool:
    """True when both external OCR binaries are on PATH."""

    return bool(shutil.which("pdftoppm")) and bool(shutil.which("tesseract"))


def extract_text_layer(pdf_bytes: bytes) -> str:
    """Text already embedded in the PDF. Empty string for a pure scan."""

    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf is not installed, cannot read the PDF text layer")
        return ""

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as error:
        logger.warning("Could not open PDF for text extraction: %s", error)
        return ""

    pages: List[str] = []

    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as error:
            logger.warning("Text extraction failed on page %s: %s", index + 1, error)
            pages.append("")

    return "\n".join(pages).strip()


def ocr_pages(pdf_bytes: bytes, dpi: int = 300) -> str:
    """Rasterise and OCR. Only for scans - needs poppler and tesseract."""

    if not ocr_available():
        raise PdfTextUnavailable(OCR_MISSING_MESSAGE)

    import pytesseract
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes, dpi=dpi)

    pages = [
        pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 6")
        for image in images
    ]

    return "\n".join(pages).strip()


def extract(pdf_bytes: bytes, dpi: int = 300) -> str:
    """Text layer first, OCR second. Raises PdfTextUnavailable if both fail."""

    text = extract_text_layer(pdf_bytes)

    if len(text) >= MIN_TEXT_LAYER_CHARS:
        logger.info("Using the embedded PDF text layer: %s chars", len(text))
        return text

    if text:
        logger.info("Text layer is only %s chars, trying OCR for a fuller read", len(text))
    else:
        logger.info("No text layer found, falling back to OCR")

    try:
        ocr_text = ocr_pages(pdf_bytes, dpi=dpi)
    except PdfTextUnavailable:
        # A short text layer still beats nothing when OCR is unavailable.
        if text:
            logger.warning("OCR unavailable, using the short text layer instead")
            return text
        raise
    except Exception as error:
        if text:
            logger.warning("OCR failed (%s), using the short text layer instead", error)
            return text
        raise PdfTextUnavailable(f"OCR failed: {error}") from error

    if not ocr_text and not text:
        raise PdfTextUnavailable(
            "No text could be read from this PDF, by text layer or by OCR."
        )

    return ocr_text or text
