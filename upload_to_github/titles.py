"""Short click-worthy titles for clips.

The LLM normally supplies a title; these templates are used when the heuristic
fallback is active or the model forgot to return one.
"""

from __future__ import annotations

from typing import Any, Dict

# Title text colours used by ffmpeg drawtext; the band stays dark.
COLOR_VALUES: Dict[str, str] = {
    "white": "white",
    "green": "0x39FF14",
    "yellow": "0xFFEA00",
    "red": "0xFF3B30",
    "blue": "0x3DA5FF",
}

COLOR_ALIASES: Dict[str, str] = {
    "white": "white",
    "белый": "white",
    "белый цвет": "white",
    "green": "green",
    "зеленый": "green",
    "зелёный": "green",
    "yellow": "yellow",
    "желтый": "yellow",
    "жёлтый": "yellow",
    "red": "red",
    "красный": "red",
    "blue": "blue",
    "синий": "blue",
    "голубой": "blue",
}


def color(value: Any) -> str:
    """Map a user-facing colour name to an ffmpeg fontcolor value."""
    key = str(value or "").strip().lower()
    name = COLOR_ALIASES.get(key, "white")
    return COLOR_VALUES[name]


_TEMPLATES: Dict[str, str] = {
    "admiration": "ОН НЕ ОЖИДАЛ ЭТОГО",
    "delight": "ЭТО БЫЛО СЛИШКОМ СМЕШНО",
    "aggression": "ОН ПРОСТО ВЗОРВАЛСЯ",
    "shock": "ВОТ ЭТО НЕОЖИДАННО",
    "funny": "СМЕХ ДО СЛЁЗ",
    "audio_reaction": "ПОСЛУШАЙТЕ ЭТО",
    "audio": "ПОСЛУШАЙТЕ ЭТО",
    "mixed": "СМОТРИТЕ ЧТО БЫЛО",
    "fallback": "СМОТРИТЕ ЧТО СЛУЧИЛОСЬ",
}


def clean(title: Any, category: Any = "") -> str:
    """Normalise a model-generated title or fall back to a template."""
    text = str(title or "").strip().strip('"').strip("'")
    text = " ".join(text.split())
    text = text.replace("|", "").replace("\n", " ")
    if len(text) > 42:
        text = text[:42].rstrip()
    if not text:
        return _TEMPLATES.get(str(category), "СМОТРИТЕ ЧТО СЛУЧИЛОСЬ")
    return text.upper()


def fallback(category: Any = "") -> str:
    return _TEMPLATES.get(str(category), "СМОТРИТЕ ЧТО СЛУЧИЛОСЬ")
