"""Bill scraper for parliamentary legislation."""

from __future__ import annotations

import re
import time
import argparse
from typing import Any
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lib.utils.config import config


class BillScraper:
    """Scrapes bills from parliamentary websites."""

    def __init__(self):
        self.base_url = "https://www.barbadosparliament.com"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.scraping.user_agent})
        self.rate_limit_delay = config.scraping.rate_limit_delay

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException,)),
        reraise=True,
    )
    def fetch_page(self, url: str) -> str:
        """Fetch a page with retry logic."""
        print(f"Fetching: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        time.sleep(self.rate_limit_delay)

        return response.text

    def discover_bills(self) -> list[str]:
        """Discover bill URLs from the parliament website."""
        try:
            html = self.fetch_page(self.base_url + "/bills/search")
            soup = BeautifulSoup(html, "html.parser")

            bill_links = []

            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if self._is_bill_url(href):
                    full_url = self._resolve_url(href)
                    bill_links.append(full_url)

            print(f"Discovered {len(bill_links)} bill URLs")
            return bill_links

        except Exception as e:
            print(f"⚠️ Error discovering bills: {e}")
            return []

    def _is_bill_url(self, url: str) -> bool:
        """Check if URL appears to be a bill page."""
        url_lower = url.lower()
        if url_lower.startswith("/bill/") or url_lower.startswith("/bills/"):
            return True
        if "/bill/" in url_lower or "/bills/" in url_lower:
            return True
        if re.search(r"/cap-\d+\b", url_lower):
            return True
        return False

    def _resolve_url(self, url: str) -> str:
        """Resolve relative URLs to absolute URLs."""
        if url.startswith("http"):
            return url
        elif url.startswith("/"):
            return self.base_url + url
        else:
            return self.base_url + "/" + url

    def scrape_bill(self, url: str) -> dict[str, Any] | None:
        """Scrape a single bill page."""
        try:
            html = self.fetch_page(url)
            soup = BeautifulSoup(html, "html.parser")

            bill_data = {"source_url": url, "scraped_at": datetime.now().isoformat()}

            bill_data.update(self._parse_title(soup))
            bill_data.update(self._parse_bill_number(soup, url))
            bill_data.update(self._parse_status(soup))
            bill_data.update(self._parse_dates(soup))
            bill_data.update(self._parse_description(soup))

            full_text_data = self._parse_full_text(soup)
            if full_text_data.get("source_url"):
                bill_data["source_url"] = full_text_data["source_url"]
            if full_text_data.get("source_text"):
                bill_data["source_text"] = full_text_data["source_text"]

            if bill_data.get("bill_number") or bill_data.get("title"):
                print(f"✅ Scraped: {bill_data.get('bill_number', 'Unknown')}")
                return bill_data
            else:
                print(f"⚠️ Skipped (no bill number/title): {url}")
                return None

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None

    def _parse_title(self, soup: BeautifulSoup) -> dict[str, str]:
        """Parse bill title from page."""
        title_selectors = [
            "main h2",
            "h2",
            "h1.bill-title",
            "h1.legislation-title",
            "h2.bill-title",
            "h2.legislation-title",
            "h1.title",
            "h2.title",
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and len(text) > 5:
                    return {"title": text}

        return {}

    def _parse_bill_number(self, soup: BeautifulSoup, url: str) -> dict[str, str]:
        """Parse bill number from URL or page title."""
        url_lower = url.lower()

        if match := re.search(r"/details/(\d+)", url_lower):
            return {"bill_number": f"BILL-{match.group(1)}"}

        title_elem = soup.select_one("main h2") or soup.select_one("h2")
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if match := re.search(r"(.*?)\s+Bill[,.\s]", title_text, re.IGNORECASE):
                bill_name = match.group(1).strip()
                if year_match := re.search(r"(\d{4})", title_text):
                    bill_name += f" {year_match.group(1)}"
                return {"bill_number": bill_name.upper()}

        if match := re.search(r"/(?:bill|legislation)/([a-z0-9-]+)\b", url_lower):
            return {"bill_number": match.group(1).upper()}

        if match := re.search(r"/cap-(\d+)\b", url_lower):
            return {"bill_number": f"Cap {match.group(1)}"}

        page_text = soup.get_text(" ")
        if match := re.search(r"\b(?:HR|SB|HB)\s*-?\s*\d+\b", page_text, re.IGNORECASE):
            return {"bill_number": match.group(0).replace(" ", "").upper()}

        return {}

    def _parse_status(self, soup: BeautifulSoup) -> dict[str, str]:
        """Parse bill status from page."""
        table = soup.select_one("table")
        if table:
            for row in table.select("tr"):
                th = row.select_one("th")
                td = row.select_one("td")
                if th and td:
                    th_text = th.get_text(strip=True).lower()
                    td_text = td.get_text(strip=True)
                    if "current stage" in th_text or "stage" in th_text:
                        return {"status": td_text}

        status_keywords = {
            "introduced": "Introduced",
            "first reading": "First Reading",
            "second reading": "Second Reading",
            "third reading": "Third Reading",
            "passed": "Passed",
            "assented": "Assented",
            "rejected": "Rejected",
            "withdrawn": "Withdrawn",
            "house of assembly": "House of Assembly",
            "senate": "Senate",
        }

        page_text = soup.get_text().lower()

        for keyword, status in status_keywords.items():
            if keyword in page_text:
                return {"status": status}

        return {"status": "Unknown"}

    def _parse_dates(self, soup: BeautifulSoup) -> dict[str, str]:
        """Parse bill dates (introduced, passed) from table."""
        table = soup.select_one("table")
        if table:
            for row in table.select("tr"):
                th = row.select_one("th")
                td = row.select_one("td")
                if th and td:
                    th_text = th.get_text(strip=True).lower()
                    td_text = td.get_text(strip=True)
                    if "notice date" in th_text:
                        return {"introduced_date": self._convert_date(td_text)}

        date_patterns = [
            (r"introduced[:\s]*(\d{1,2}[-\s](\w+)[-,\s](\d{4}))", "introduced_date"),
            (r"passed[:\s]*(\d{1,2}[-\s](\w+)[-,\s](\d{4}))", "passed_date"),
            (r"submitted[:\s]*(\d{1,2}[-\s](\w+)[-,\s](\d{4}))", "introduced_date"),
            (r"assented[:\s]*(\d{1,2}[-\s](\w+)[-,\s](\d{4}))", "passed_date"),
        ]

        dates = {}
        page_text = soup.get_text()

        for pattern, field in date_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                dates[field] = self._convert_date(match.group(1))

        return dates

    def _convert_date(self, date_str: str) -> str:
        """Convert DD/MM/YYYY to YYYY-MM-DD for PostgreSQL."""
        date_str = date_str.strip()
        parts = date_str.split("/")
        if len(parts) == 3:
            day, month, year = parts
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return date_str

    def _parse_description(self, soup: BeautifulSoup) -> dict[str, str]:
        """Parse bill description from page."""
        desc_selectors = [
            "div.description",
            "div.bill-description",
            "div.legislation-description",
            "div.summary",
            "p.summary",
            "p.description",
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                desc = element.get_text(strip=True)
                if len(desc) > 50:
                    return {"description": desc}

        return {}

    def _parse_full_text(self, soup: BeautifulSoup) -> dict[str, str]:
        """Parse full bill text or PDF link from page."""
        text_selectors = [
            "div.bill-text",
            "div.legislation-text",
            "div.full-text",
            "article.bill-content",
            "main.bill-content",
        ]

        for selector in text_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if len(text) > 100:
                    return {"source_text": text}

        pdf_link = soup.select_one("a[href$='.pdf']")
        if pdf_link:
            href = pdf_link.get("href", "")
            if href and not href.startswith("http"):
                href = self.base_url + href
            if href:
                return {"source_url": href}

        return {}

    def scrape_all_bills(self, max_bills: int | None = None) -> list[dict[str, Any]]:
        """Scrape all discovered bills."""
        bill_urls = self.discover_bills()

        if not bill_urls:
            print("❌ No bills discovered")
            return []

        if max_bills:
            bill_urls = bill_urls[:max_bills]

        print(f"Scraping {len(bill_urls)} bills...")

        bills = []
        for i, url in enumerate(bill_urls, 1):
            print(f"\n[{i}/{len(bill_urls)}] Processing: {url}")

            bill_data = self.scrape_bill(url)

            if bill_data:
                bills.append(bill_data)

            if max_bills and len(bills) >= max_bills:
                print(f"\n✅ Reached max bills limit: {max_bills}")
                break

        print(f"\n✅ Successfully scraped {len(bills)} bills")
        return bills


def main():
    parser = argparse.ArgumentParser(description="Scrape bills from parliamentary website")
    parser.add_argument(
        "--max-bills",
        type=int,
        default=None,
        help="Maximum number of bills to scrape (default: all)",
    )
    parser.add_argument(
        "--output-file",
        default="bills_scraped.json",
        help="Output JSON file for scraped bills",
    )
    parser.add_argument(
        "--source-url",
        default="https://www.barbadosparliament.com",
        help="Base URL for bill discovery",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Bill Scraper - Phase 2: Bill Scraping Pipeline")
    print("=" * 80)
    print(f"Source URL: {args.source_url}")
    print(f"Max Bills: {args.max_bills or 'All'}")
    print(f"Output File: {args.output_file}")
    print("=" * 80)

    scraper = BillScraper()
    bills = scraper.scrape_all_bills(max_bills=args.max_bills)

    import json

    with open(args.output_file, "w") as f:
        json.dump(bills, f, indent=2)

    print(f"\n✅ Saved {len(bills)} bills to {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
