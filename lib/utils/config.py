"""Configuration management for parliamentary search system."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration."""

    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_database: str = os.getenv("POSTGRES_DATABASE", "parliament_search")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    # Provider: "google_ai" (AI Studio) or "vertex_ai" (Vertex AI).
    provider: str = os.getenv("EMBEDDING_PROVIDER", "google_ai")

    # AI Studio key
    api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Vertex AI settings (requires ADC credentials or service account)
    vertex_project: str = os.getenv("VERTEX_PROJECT", "")
    vertex_location: str = os.getenv("VERTEX_LOCATION", "")

    # google-genai model ids change over time; text-embedding-004 was a common default,
    # but availability depends on your project/key.
    model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))


@dataclass
class VideoConfig:
    """Video processing configuration."""

    youtube_api_key: str | None = os.getenv("YOUTUBE_API_KEY")
    segment_minutes: int = int(os.getenv("SEGMENT_MINUTES", "30"))
    overlap_minutes: int = int(os.getenv("OVERLAP_MINUTES", "1"))


@dataclass
class ScrapingConfig:
    """Bill scraping configuration."""

    rate_limit_delay: float = float(os.getenv("SCRAPE_RATE_LIMIT", "1.0"))
    user_agent: str = os.getenv(
        "SCRAPE_USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    )


@dataclass
class AppConfig:
    """Application configuration."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    chat_trace: bool = os.getenv("CHAT_TRACE", "").lower() in {"1", "true", "on"}
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "").lower() in {
        "1",
        "true",
        "on",
    }
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


config = AppConfig()
