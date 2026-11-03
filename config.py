"""
config.py — one place for every setting.

All values come from the .env file (or the environment). Nothing secret is
ever written in the code. If a required value is missing, a clear
ConfigError is raised so the app can show a friendly message instead of
crashing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class ConfigError(Exception):
    """Raised when a setting is missing or invalid."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got '{raw}'.") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got '{raw}'.") from exc


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    api_key: str
    model: str
    base_url: str | None          # set only for OpenAI-compatible providers (Groq, Ollama, Gemini...)
    temperature: float
    request_timeout: int          # seconds per LLM call
    max_retries: int              # automatic retries on network / rate-limit errors
    ticket_timeout: int           # seconds for the whole 4-agent run
    max_messages: int             # safety net for the team conversation

    # --- RAG / FAISS ---
    embedding_provider: str       # "openai" or "local"
    embedding_model: str
    top_k: int
    min_similarity: float

    # --- Paths ---
    kb_dir: Path
    index_dir: Path
    log_file: Path

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("your-")


def get_settings() -> Settings:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "local"}:
        raise ConfigError("EMBEDDING_PROVIDER must be 'openai' or 'local'.")

    # Similarity scores are on a different scale for each embedding type.
    default_min_sim = 0.30 if provider == "openai" else 0.18

    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        base_url=(os.getenv("OPENAI_BASE_URL", "").strip() or None),
        temperature=_float("TEMPERATURE", 0.2),
        request_timeout=_int("REQUEST_TIMEOUT", 60),
        max_retries=_int("MAX_RETRIES", 3),
        ticket_timeout=_int("TICKET_TIMEOUT", 180),
        max_messages=_int("MAX_MESSAGES", 10),
        embedding_provider=provider,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        top_k=_int("TOP_K", 3),
        min_similarity=_float("MIN_SIMILARITY", default_min_sim),
        kb_dir=BASE_DIR / os.getenv("KB_DIR", "knowledge_base"),
        index_dir=BASE_DIR / os.getenv("INDEX_DIR", "vector_store"),
        log_file=BASE_DIR / os.getenv("LOG_FILE", "logs/tickets.csv"),
    )


def require_api_key(settings: Settings) -> None:
    if not settings.has_api_key:
        raise ConfigError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and paste your key "
            "after OPENAI_API_KEY= (no quotes, no spaces)."
        )
