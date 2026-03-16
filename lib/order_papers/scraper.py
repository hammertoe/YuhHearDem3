from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

from lib.utils.config import config


@dataclass(frozen=True)
class OrderPaperAttachment:
    label: str
    url: str


@dataclass(frozen=True)
class OrderPaperEntry:
    title: str
    posted_date: str
    pdf_url: str
    attachments: list[OrderPaperAttachment]
    chamber: str | None = None


def parse_order_paper_search_html(
    html: str, *, chamber: str | None = None
) -> list[OrderPaperEntry]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.modern-style")
    entries: list[OrderPaperEntry] = []

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        primary_link = None
        for link in cells[0].find_all("a", href=True):
            href = str(link.get("href", ""))
            if href.lower().endswith(".pdf"):
                primary_link = link
                break

        if not primary_link:
            continue

        title = primary_link.get_text(strip=True)
        pdf_url = str(primary_link.get("href", "")).strip()
        posted_date = cells[1].get_text(strip=True)

        attachments = _extract_attachments(cells[0], primary_url=pdf_url)

        entries.append(
            OrderPaperEntry(
                title=title,
                posted_date=posted_date,
                pdf_url=pdf_url,
                attachments=attachments,
                chamber=chamber,
            )
        )

    return entries


def _extract_attachments(cell: BeautifulSoup, *, primary_url: str) -> list[OrderPaperAttachment]:
    attachments: list[OrderPaperAttachment] = []
    seen = {primary_url}

    for link in cell.find_all("a", href=True):
        href = str(link.get("href", "")).strip()
        if not href.lower().endswith(".pdf"):
            continue
        if href in seen:
            continue

        label = link.get_text(strip=True)
        attachments.append(OrderPaperAttachment(label=label, url=href))
        seen.add(href)

    return attachments


class OrderPaperScraper:
    def __init__(self, base_url: str = "https://www.barbadosparliament.com") -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.scraping.user_agent})
        self.rate_limit_delay = config.scraping.rate_limit_delay

    def fetch_search_page(self, *, chamber: str, offset: int = 0, keyword: str = "") -> str:
        chamber_type = _resolve_chamber_type(chamber)
        url = f"{self.base_url}/order_papers/search/type/{chamber_type}"

        if offset <= 0:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text

        data = {
            "OP_KEYWORD_S": keyword,
            "COL_ID": "",
            "ORD_ID": "",
            "OFF_SET": str(offset),
        }
        response = self.session.post(url, data=data, timeout=30)
        response.raise_for_status()
        return response.text

    def list_order_papers(
        self,
        *,
        chamber: str,
        max_pages: int = 1,
        keyword: str = "",
        page_size: int = 20,
    ) -> list[OrderPaperEntry]:
        entries: list[OrderPaperEntry] = []
        for page_index in range(max_pages):
            offset = page_index * page_size
            html = self.fetch_search_page(chamber=chamber, offset=offset, keyword=keyword)
            page_entries = parse_order_paper_search_html(html, chamber=chamber)
            if not page_entries:
                break
            entries.extend(page_entries)
        return entries


def resolve_order_paper_urls(
    entries: Iterable[OrderPaperEntry], base_url: str
) -> list[OrderPaperEntry]:
    resolved: list[OrderPaperEntry] = []
    for entry in entries:
        pdf_url = urljoin(base_url, entry.pdf_url)
        attachments = [
            OrderPaperAttachment(label=a.label, url=urljoin(base_url, a.url))
            for a in entry.attachments
        ]
        resolved.append(
            OrderPaperEntry(
                title=entry.title,
                posted_date=entry.posted_date,
                pdf_url=pdf_url,
                attachments=attachments,
                chamber=entry.chamber,
            )
        )
    return resolved


def build_filename_from_url(url: str) -> str:
    name = url.split("?")[0].split("#")[0].rstrip("/")
    filename = name.rsplit("/", 1)[-1]
    return unquote(filename) or "order_paper.pdf"


def _resolve_chamber_type(chamber: str) -> int:
    chamber_value = chamber.strip().lower()
    if chamber_value == "house":
        return 1
    if chamber_value == "senate":
        return 2
    raise ValueError(f"Unsupported chamber: {chamber}")
