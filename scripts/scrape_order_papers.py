#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.order_papers.scraper import (  # noqa: E402
    OrderPaperEntry,
    OrderPaperScraper,
    build_filename_from_url,
    resolve_order_paper_urls,
)
from scripts.ingest_order_paper_pdf import ingest_order_paper  # noqa: E402


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(session: requests.Session, url: str, output_path: Path) -> None:
    if output_path.exists():
        return
    response = session.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _collect_entries(
    scraper: OrderPaperScraper, chamber: str, max_pages: int
) -> list[OrderPaperEntry]:
    entries = scraper.list_order_papers(chamber=chamber, max_pages=max_pages)
    return resolve_order_paper_urls(entries, scraper.base_url)


def _download_entry(
    session: requests.Session,
    entry: OrderPaperEntry,
    output_dir: Path,
    include_attachments: bool,
) -> list[tuple[Path, str | None]]:
    downloaded: list[tuple[Path, str | None]] = []
    filename = build_filename_from_url(entry.pdf_url)
    target = output_dir / filename
    _download_file(session, entry.pdf_url, target)
    downloaded.append((target, entry.chamber))

    if include_attachments:
        for attachment in entry.attachments:
            attachment_name = build_filename_from_url(attachment.url)
            attachment_target = output_dir / attachment_name
            _download_file(session, attachment.url, attachment_target)
            downloaded.append((attachment_target, entry.chamber))

    return downloaded


def _ingest_files(paths: list[tuple[Path, str | None]]) -> list[str]:
    ingested: list[str] = []
    for path, chamber in paths:
        order_paper_id = ingest_order_paper(str(path), chamber or "auto")
        ingested.append(order_paper_id)
    return ingested


def _write_manifest(output_dir: Path, entries: list[OrderPaperEntry]) -> Path:
    payload = [
        {
            "title": entry.title,
            "posted_date": entry.posted_date,
            "pdf_url": entry.pdf_url,
            "attachments": [
                {"label": attachment.label, "url": attachment.url}
                for attachment in entry.attachments
            ],
            "chamber": entry.chamber,
        }
        for entry in entries
    ]
    manifest_path = output_dir / "order_papers_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Barbados order papers and ingest PDFs")
    parser.add_argument(
        "--chamber",
        choices=["house", "senate", "both"],
        default="both",
        help="Which chamber to scrape (default: both)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages per chamber to scrape (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/order_papers",
        help="Directory to save downloaded PDFs (default: data/order_papers)",
    )
    parser.add_argument(
        "--include-attachments",
        action="store_true",
        help="Download booklet/supplement attachments",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest downloaded PDFs into the database",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    scraper = OrderPaperScraper()
    session = scraper.session

    entries: list[OrderPaperEntry] = []
    if args.chamber in {"house", "both"}:
        entries.extend(_collect_entries(scraper, "house", args.max_pages))
    if args.chamber in {"senate", "both"}:
        entries.extend(_collect_entries(scraper, "senate", args.max_pages))

    if not entries:
        print("✅ No order papers found")
        return 0

    manifest_path = _write_manifest(output_dir, entries)
    print(f"✅ Saved manifest: {manifest_path}")

    all_downloaded: list[tuple[Path, str | None]] = []
    for entry in entries:
        downloaded = _download_entry(
            session,
            entry,
            output_dir,
            include_attachments=args.include_attachments,
        )
        all_downloaded.extend(downloaded)

    print(f"✅ Downloaded {len(all_downloaded)} PDF(s) to {output_dir}")

    if args.ingest:
        ingested_ids = _ingest_files(all_downloaded)
        print(f"✅ Ingested {len(ingested_ids)} order paper PDF(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
