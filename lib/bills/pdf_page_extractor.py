"""PDF page extraction for bill documents using Gemini vision."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from lib.google_client import GeminiClient


@dataclass(frozen=True)
class ExtractedPdfPage:
    page_number: int
    text: str


class BillPdfPageExtractor:
    """Extract per-page text from bill PDFs using Gemini 2.5 Flash."""

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self.gemini_client = gemini_client or GeminiClient(model="gemini-2.5-flash")

    def extract_pages(self, pdf_url: str) -> list[ExtractedPdfPage]:
        url = str(pdf_url or "").strip()
        if not url.lower().endswith(".pdf"):
            return []

        local_pdf = self._download_pdf(url)
        if local_pdf is None:
            return []

        try:
            parsed = self.gemini_client.analyze_pdf_with_vision(
                pdf_path=local_pdf,
                prompt=self._prompt(),
                response_schema=self._schema(),
            )
        except Exception as exc:
            print(f"⚠️ Failed to extract PDF pages from {url}: {exc}")
            return []
        finally:
            try:
                local_pdf.unlink(missing_ok=True)
            except Exception:
                pass

        raw_pages = parsed.get("pages") if isinstance(parsed, dict) else None
        if not isinstance(raw_pages, list):
            return []

        out: list[ExtractedPdfPage] = []
        seen: set[int] = set()
        for item in raw_pages:
            if not isinstance(item, dict):
                continue
            page_number = int(item.get("page_number") or 0)
            text = str(item.get("text") or "").strip()
            if page_number <= 0 or not text or page_number in seen:
                continue
            seen.add(page_number)
            out.append(ExtractedPdfPage(page_number=page_number, text=text))

        out.sort(key=lambda p: p.page_number)
        return out

    def _download_pdf(self, pdf_url: str) -> Path | None:
        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
        except Exception as exc:
            print(f"⚠️ Failed to download PDF {pdf_url}: {exc}")
            return None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            return Path(tmp.name)

    def _prompt(self) -> str:
        return (
            "Extract text from this Barbados parliamentary bill PDF page by page. "
            "Return JSON with a `pages` array. "
            "For each page, include: "
            "- page_number: 1-based page number as an integer\n"
            "- text: plain text content from that page\n\n"
            "Rules:\n"
            "- Preserve page boundaries exactly.\n"
            "- Keep the original wording; do not summarize.\n"
            "- Remove headers/footers only if they are obvious repetitive boilerplate.\n"
            "- If a page is mostly blank, include an empty string for text."
        )

    def _schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page_number": {"type": "integer"},
                            "text": {"type": "string"},
                        },
                        "required": ["page_number", "text"],
                    },
                }
            },
            "required": ["pages"],
        }
