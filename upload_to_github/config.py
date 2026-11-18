"""Configuration for the clipper bot.

All settings can be overridden with environment variables so the bot can run
24/7 on a machine without editing code.  Everything runs locally: no paid API,
no external AI service.  The only network requests are to YouTube (yt-dlp) and
the one-time download of a Whisper model from HuggingFace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _box_env(name: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse "x,y,w,h" as fractions of the frame, e.g. 0.02,0.05,0.26,0.29."""
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        values = tuple(float(part) for part in raw.replace(";", ",").split(","))
        if len(values) == 4:
            return values  # type: ignore[return-value]
    except ValueError:
        pass
    return None


@dataclass
class Config:
    # Telegram
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_proxy: str = field(default_factory=lambda: os.getenv("TELEGRAM_PROXY", ""))

    # Paths
    work_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CLIPPER_WORK_DIR", "work"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CLIPPER_OUTPUT_DIR", "output"))
    )

    # Whisper (local, free)
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )

    # Clipping
    min_clip: int = field(default_factory=lambda: _int_env("MIN_CLIP", 15))
    max_clip: int = field(default_factory=lambda: _int_env("MAX_CLIP", 60))
    max_clips: int = field(default_factory=lambda: _int_env("MAX_CLIPS", 5))

    # Presentation
    subtitles: bool = field(default_factory=lambda: _bool_env("SUBTITLES", True))
    layout: str = field(
        default_factory=lambda: os.getenv("LAYOUT", "top_webcam_bottom_screen")
    )
    title_color: str = field(
        default_factory=lambda: os.getenv("TITLE_COLOR", "white")
    )

    # Auto-monitor
    monitor_interval_min: int = field(default_factory=lambda: _int_env("MONITOR_INTERVAL", 10))
    monitor_tail_seconds: int = field(default_factory=lambda: _int_env("MONITOR_TAIL", 180))
    monitor_max_clips: int = field(default_factory=lambda: _int_env("MONITOR_MAX_CLIPS", 2))

    # Local LLM (Ollama) — semantic moment detection
    llm_enabled: bool = field(default_factory=lambda: _bool_env("LLM_ENABLED", True))
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    )
    llm_timeout: int = field(default_factory=lambda: _int_env("LLM_TIMEOUT", 180))
    llm_chunk_seconds: int = field(default_factory=lambda: _int_env("LLM_CHUNK_SECONDS", 600))
    llm_max_chunks: int = field(default_factory=lambda: _int_env("LLM_MAX_CHUNKS", 24))
    llm_fallback: bool = field(default_factory=lambda: _bool_env("LLM_FALLBACK", True))
    llm_global_rank: bool = field(
        default_factory=lambda: _bool_env("LLM_GLOBAL_RANK", True)
    )

    # Visual reaction check on selected clips only (cheap Haar cascades).
    visual_enabled: bool = field(
        default_factory=lambda: _bool_env("VISUAL_ENABLED", True)
    )
    visual_samples: int = field(default_factory=lambda: _int_env("VISUAL_SAMPLES", 10))
    visual_min_score: float = field(
        default_factory=lambda: _float_env("VISUAL_MIN_SCORE", 0.3)
    )

    # Manual webcam rectangle as frame fractions x,y,w,h (0..1). None = auto.
    webcam_box: Optional[Tuple[float, float, float, float]] = field(
        default_factory=lambda: _box_env("WEBCAM_BOX")
    )

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.output_dir = Path(self.output_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def copy(self, **overrides: object) -> "Config":
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}
        data.update(overrides)
        return Config(**data)  # type: ignore[arg-type]


DEFAULT_CONFIG = Config()
