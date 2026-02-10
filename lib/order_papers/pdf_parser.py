"""Order paper PDF parser using Gemini vision."""

from datetime import datetime

from lib.order_papers.models import AgendaItem, OrderPaper, OrderPaperSpeaker
from lib.google_client import GeminiClient


class OrderPaperParser:
    """Parse parliamentary order paper PDFs to extract speakers and agenda."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        """Initialize parser with Gemini client.

        Args:
            gemini_client: Initialized Gemini client instance
        """
        self.client = gemini_client

    def parse(self, pdf_path: str) -> OrderPaper:
        """Parse an order paper PDF.

        Note: PDF pages may be arranged for printing and not in logical reading
        order. The parser handles this by having Gemini understand the document
        structure holistically.

        Args:
            pdf_path: Path to order paper PDF

        Returns:
            Parsed OrderPaper object with speakers and agenda items
        """
        prompt = self._build_extraction_prompt()
        schema = self._build_response_schema()

        response = self.client.analyze_pdf_with_vision(
            pdf_path=pdf_path,
            prompt=prompt,
            response_schema=schema,
        )

        return self._parse_response(response)

    def _build_extraction_prompt(self) -> str:
        """Build extraction prompt for Gemini to extract order paper information.

        Returns:
            Detailed extraction prompt
        """
        return """Analyze this Barbados parliamentary order paper PDF and extract the following information.

IMPORTANT: The PDF pages may be arranged for printing and not in logical reading order.
Please read and understand the entire document structure before extracting information.

Extract:

1. **Session Information:**
   - The full session title (e.g., "The Honourable The Senate, First Session of 2022-2027")
   - The sitting number (e.g., "Sixty-Seventh Sitting")
   - The session date in YYYY-MM-DD format

2. **All Speakers/Senators:**
   - Extract EVERY unique person mentioned in the document
   - For each person, extract:
     - Full name (e.g., "L. R. Cummins", "J. X. Walcott")
     - Title (e.g., "Hon.", "Dr.", "Rev.", "Most Honourable")
     - Role/position if mentioned (e.g., "Minister of Finance", "Senator", "President")
   - Look in committee memberships, agenda items, and any other sections
   - Deduplicate speakers - if someone appears multiple times, only include them once with their most complete information

3. **Agenda Items:**
   - Extract all items from "Public Business" and "Private Members' Business" sections
   - For each item:
     - Topic title (e.g., "Cybercrime Bill, 2024")
     - Primary speaker (who is moving/presenting it)
     - Brief description if available

Return the information in the specified JSON structure."""

    def _build_response_schema(self) -> dict:
        """Build JSON schema for structured response.

        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "session_title": {
                    "type": "string",
                    "description": "Full session title",
                },
                "sitting_number": {
                    "type": "string",
                    "description": "Sitting number (e.g., 'Sixty-Seventh Sitting')",
                },
                "session_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Session date in YYYY-MM-DD format",
                },
                "speakers": {
                    "type": "array",
                    "description": "All unique speakers/senators mentioned",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "title": {"type": "string"},
                            "role": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
                "agenda_items": {
                    "type": "array",
                    "description": "All agenda items to be discussed",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic_title": {"type": "string"},
                            "primary_speaker": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["topic_title"],
                    },
                },
            },
            "required": ["session_title", "session_date", "speakers", "agenda_items"],
        }

    def _parse_response(self, response: dict) -> OrderPaper:
        """Parse Gemini response into OrderPaper object.

        Args:
            response: JSON response from Gemini

        Returns:
            OrderPaper object
        """
        session_date = datetime.strptime(response["session_date"], "%Y-%m-%d").date()

        speakers = [
            OrderPaperSpeaker(
                name=s["name"],
                title=s.get("title"),
                role=s.get("role"),
            )
            for s in response.get("speakers", [])
        ]

        agenda_items = [
            AgendaItem(
                topic_title=item["topic_title"],
                primary_speaker=item.get("primary_speaker"),
                description=item.get("description"),
            )
            for item in response.get("agenda_items", [])
        ]

        return OrderPaper(
            session_title=response["session_title"],
            session_date=session_date,
            sitting_number=response.get("sitting_number"),
            speakers=speakers,
            agenda_items=agenda_items,
        )
