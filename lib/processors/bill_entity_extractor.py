"""Bill entity extraction for processing scraped bills."""

from __future__ import annotations

import re
import argparse
import json
from typing import Any

from rapidfuzz import fuzz

from lib.id_generators import generate_bill_id


class BillEntityExtractor:
    """Extract entities from bill text using regex patterns."""

    def extract_entities_from_bill(self, bill_data: dict[str, Any]) -> dict[str, Any]:
        """Extract entities from bill text."""
        description = bill_data.get("description", "")
        source_text = bill_data.get("source_text", "")

        combined_text = f"{description} {source_text}"

        entities = {
            "topics": [],
            "organizations": [],
            "persons": [],
            "dates": [],
            "locations": [],
            "related_bills": [],
        }

        entities["organizations"] = self._extract_organizations(combined_text)
        entities["persons"] = self._extract_persons(combined_text)
        entities["locations"] = self._extract_locations(combined_text)
        entities["dates"] = self._extract_dates(combined_text)
        entities["related_bills"] = self._extract_related_bills(combined_text)
        entities["topics"] = self._extract_topics(combined_text, entities)

        bill_data["extracted_entities"] = entities
        return bill_data

    def _extract_organizations(self, text: str) -> list[dict[str, Any]]:
        """Extract organization names using regex patterns."""
        organizations = []

        patterns = [
            r"\b(?:Ministry|Department|Office|Agency|Commission|Authority|Council|Board)\s+of\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
            r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Ministry|Department|Office|Agency|Commission|Authority|Council|Board)\b",
            r"\b(?:The\s+)?[A-Z]{2,}(?:\s+[A-Z]{2,})*\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                org_text = match.group(0).strip()
                organizations.append(
                    {
                        "text": org_text,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return organizations

    def _extract_persons(self, text: str) -> list[dict[str, Any]]:
        """Extract person names using regex patterns."""
        persons = []

        patterns = [
            r"\b(?:Mr|Mrs|Ms|Dr|Hon|Prof)\.?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
            r"\b[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                person_text = match.group(0).strip()
                persons.append(
                    {
                        "text": person_text,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return persons

    def _extract_locations(self, text: str) -> list[dict[str, Any]]:
        """Extract locations using regex patterns."""
        locations = []

        patterns = [
            r"\b(?:Jamaica|Kingston|Portmore|Spanish\s+Town|Montego\s+Bay|Mandeville|May\s+Pen|Ocho\s+Rios)\b",
            r"\b(?:St\.|Saint\s+)(?:Andrew|Catherine|Thomas|Mary|Ann|James|Elizabeth|Mary)\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                loc_text = match.group(0).strip()
                locations.append(
                    {
                        "text": loc_text,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return locations

    def _extract_dates(self, text: str) -> list[dict[str, Any]]:
        """Extract dates using regex patterns."""
        dates = []

        patterns = [
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                date_text = match.group(0).strip()
                dates.append(
                    {
                        "text": date_text,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return dates

    def _extract_related_bills(self, text: str) -> list[dict[str, Any]]:
        """Extract related bill/act references."""
        bills = []

        patterns = [
            r"\b(?:the\s+)?[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+Act\b",
            r"\b(?:the\s+)?[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+Bill\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                bill_text = match.group(0).strip()
                if bill_text.lower() not in {"this bill", "the bill", "a bill"}:
                    bills.append(
                        {
                            "text": bill_text,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

        return bills

    def _extract_topics(
        self, text: str, entities: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, str]]:
        """Extract topics using noun phrases."""
        topics = []

        patterns = [
            r"\b(?:[A-Z][a-z]+(?:\s+[a-z]+){0,3})\s+(?:Act|Bill|Law|Regulation|Policy|Program|Scheme|Fund|System|Authority|Commission)\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                topic_text = match.group(0).strip()
                if len(topic_text) > 3 and len(topic_text) < 50:
                    if not self._is_entity(topic_text, entities):
                        topics.append(
                            {
                                "text": topic_text,
                                "start": match.start(),
                                "end": match.end(),
                            }
                        )

        return topics[:10]

    def _is_entity(self, text: str, entities: dict[str, list[dict[str, Any]]]) -> bool:
        """Check if text matches any known entity."""
        for entity_list in entities.values():
            for entity in entity_list:
                if fuzz.ratio(text.lower(), entity["text"].lower()) > 90:
                    return True
        return False

    def extract_category(self, bill_data: dict[str, Any]) -> str:
        """Categorize bill based on entities."""
        entities = bill_data.get("extracted_entities", {})

        keywords = " ".join(
            [e["text"] for e in entities.get("organizations", [])[:5]]
            + [e["text"] for e in entities.get("topics", [])[:5]]
        )

        keywords_lower = keywords.lower()

        category_keywords = {
            "Transport": [
                "transport",
                "road",
                "traffic",
                "vehicle",
                "highway",
                "driving",
            ],
            "Health": ["health", "medical", "hospital", "clinic", "nurse", "doctor"],
            "Finance": ["finance", "budget", "tax", "revenue", "financial", "money"],
            "Education": [
                "education",
                "school",
                "university",
                "college",
                "student",
                "teacher",
            ],
            "Justice": ["justice", "law", "court", "legal", "crime", "police"],
            "Agriculture": ["agriculture", "farm", "fishing", "food", "crop"],
            "Housing": ["housing", "home", "apartment", "rent", "building"],
            "Environment": ["environment", "climate", "energy", "pollution", "waste"],
            "Labor": ["labor", "worker", "employment", "wage", "job"],
        }

        for category, keywords_list in category_keywords.items():
            for keyword in keywords_list:
                if keyword in keywords_lower:
                    return category

        return "General"

    def process_bills(self, bills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process bills with entity extraction and categorization."""
        print(f"\nProcessing {len(bills)} bills...")

        processed_bills = []
        existing_bill_ids = set()

        for i, bill in enumerate(bills, 1):
            print(
                f"[{i}/{len(bills)}] Processing: {bill.get('bill_number', 'Unknown')}"
            )

            bill = self.extract_entities_from_bill(bill)

            bill_id = generate_bill_id(bill.get("bill_number", ""), existing_bill_ids)
            existing_bill_ids.add(bill_id)
            bill["id"] = bill_id

            bill["category"] = self.extract_category(bill)

            bill["keywords"] = self._generate_keywords(bill)

            processed_bills.append(bill)

        print(f"\n✅ Processed {len(processed_bills)} bills")
        return processed_bills

    def _generate_keywords(self, bill: dict[str, Any]) -> list[str]:
        """Generate keywords from bill title and entities."""
        stopwords = {
            "the",
            "and",
            "or",
            "of",
            "to",
            "for",
            "in",
            "on",
            "by",
            "with",
            "a",
            "an",
            "be",
            "it",
            "this",
            "that",
        }

        def tokenize(text: str) -> list[str]:
            tokens = [t for t in re.findall(r"\b\w+\b", text.lower()) if len(t) >= 3]
            return [t for t in tokens if t not in stopwords]

        ordered: list[str] = []
        seen: set[str] = set()

        for word in tokenize(bill.get("title", "")):
            if word not in seen:
                ordered.append(word)
                seen.add(word)

        entities = bill.get("extracted_entities", {})
        for entity_list in entities.values():
            for entity in entity_list:
                for word in tokenize(str(entity.get("text", ""))):
                    if word not in seen:
                        ordered.append(word)
                        seen.add(word)

        return ordered[:15]


def main():
    parser = argparse.ArgumentParser(description="Extract entities from scraped bills")
    parser.add_argument(
        "--input-file",
        default="bills_scraped.json",
        help="Input JSON file with scraped bills",
    )
    parser.add_argument(
        "--output-file",
        default="bills_processed.json",
        help="Output JSON file for processed bills",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Bill Entity Extractor - Phase 2: Entity Extraction")
    print("=" * 80)
    print(f"Input File: {args.input_file}")
    print(f"Output File: {args.output_file}")
    print("=" * 80)

    with open(args.input_file, "r") as f:
        bills = json.load(f)

    extractor = BillEntityExtractor()
    processed_bills = extractor.process_bills(bills)

    with open(args.output_file, "w") as f:
        json.dump(processed_bills, f, indent=2)

    print(f"\n✅ Saved {len(processed_bills)} processed bills to {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
