"""Gemini API client wrapper for vision and text generation."""

import json
import os
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig
from typing import Any


class GeminiClient:
    """Wrapper for Google Gemini API operations."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model name to use (default: gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be provided or set in environment")
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def analyze_pdf_with_vision(
        self,
        pdf_path: Path,
        prompt: str,
        response_schema: dict | None = None,
    ) -> dict[str, Any]:
        """Analyze a PDF using Gemini vision with optional structured output.

        Args:
            pdf_path: Path to PDF file
            prompt: Instruction prompt for analysis
            response_schema: Optional JSON schema for structured output

        Returns:
            Parsed JSON response from model
        """
        uploaded_file = self.client.files.upload(file=str(pdf_path))

        config_kwargs = {
            "temperature": 0.0,
            "max_output_tokens": 65536,
        }
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        generation_config = GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, uploaded_file],
            config=generation_config,
        )

        return self._safe_json_parse(response.text or "", context="PDF vision analysis")

    def _safe_json_parse(self, response_text: str, context: str = "") -> dict[str, Any]:
        """Parse JSON response with error handling.

        Args:
            response_text: Raw JSON string from Gemini API
            context: Context string for error messages (e.g., "entity extraction")

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: With enhanced error message showing response preview
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            preview = response_text[:200] if response_text else ""
            raise ValueError(
                f"Failed to parse JSON response ({context}). Preview: {preview}..."
            )
