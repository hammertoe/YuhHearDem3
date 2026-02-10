"""Bill entity extraction for processing scraped bills."""

from __future__ import annotations

import re
import argparse
import json
from typing import Any

import spacy
from rapidfuzz import fuzz

from lib.id_generators import generate_bill_id


class BillEntityExtractor:
    """Extract entities from bill text."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def extract_entities_from_bill(self, bill_data: dict[str, Any]) -> dict[str, Any]:
        """Extract entities from bill text."""
        source_text = bill_data.get("source_text", "")
        description = bill_data.get("description", "")

        combined_text = f"{description} {source_text}"

        doc = self.nlp(combined_text)

        entities = {
            "topics": [],
            "organizations": [],
            "persons": [],
            "dates": [],
            "locations": [],
            "related_bills": [],
        }

        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text

            if entity_type == "ORG":
                entities["organizations"].append(
                    {"text": entity_text, "start": ent.start_char, "end": ent.end_char}
                )
            elif entity_type == "PERSON":
                entities["persons"].append(
                    {"text": entity_text, "start": ent.start_char, "end": ent.end_char}
                )
            elif entity_type == "GPE" or entity_type == "LOC":
                entities["locations"].append(
                    {"text": entity_text, "start": ent.start_char, "end": ent.end_char}
                )
            elif entity_type == "DATE":
                entities["dates"].append(
                    {"text": entity_text, "start": ent.start_char, "end": ent.end_char}
                )
            elif entity_type == "LAW":
                entities["related_bills"].append(
                    {"text": entity_text, "start": ent.start_char, "end": ent.end_char}
                )

        entities["topics"] = self._extract_topics(doc, entities)

        bill_data["extracted_entities"] = entities
        return bill_data

    def _extract_topics(
        self, doc: Any, entities: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, str]]:
        """Extract topics using noun chunks."""
        topics = []

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text

            if len(chunk_text) > 3 and len(chunk_text) < 50:
                if not self._is_entity(chunk_text, entities):
                    topics.append(
                        {
                            "text": chunk_text,
                            "start": chunk.start_char,
                            "end": chunk.end_char,
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
