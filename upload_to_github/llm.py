"""Client for a local Ollama model.

No API key, no internet, no paid service: Ollama runs the model on the user's
own machine and exposes an HTTP API on localhost.  We use only the Python
standard library (urllib), so this adds no new pip dependency.

Setup (once):
    install Ollama from https://ollama.com
    ollama pull qwen2.5:3b
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from config import Config

SYSTEM_PROMPT = """Ты — профессиональный монтажёр виральных коротких клипов.
Тебе даётся расшифровка стрима с таймкодами в формате [секунды] текст.

Задача: выбрать самые интересные моменты для вертикальных клипов длиной 15–60 секунд.

Правила:
1. Момент ОБЯЗАТЕЛЬНО должен начинаться с реакции: восхищение, восторг, смех, шок,
   агрессия, крик, неожиданный поворот, сильная эмоция.
2. Используй ТОЛЬКО таймкоды из расшифровки. Не выдумывай времена.
3. Длина каждого клипа от 15 до 60 секунд, start < end. Начало ставь прямо на реакцию.
4. Оценивай по СМЫСЛУ, а не только по ключевым словам: ирония, контекст, драма,
   неожиданность и реакция зрителей важнее простого произношения «вау».
5. Не выбирай скучные монологи, паузы и спокойные объяснения.
6. category — одно из: admiration, delight, aggression, shock, funny.
7. score — целое число от 1 до 10, где 10 = идеальный виральный момент.
8. reason — одно короткое предложение, почему момент интересный.
9. Строки, начинающиеся с ">>> [АУДИО", — это сильные звуковые реакции БЕЗ слов
   (крик, визг, смех, удар). Их тоже можно и нужно выбирать: используй соседние реплики
   как контекст и ставь начало клипа прямо на такой момент.
10. Каждый момент ДОЛЖЕН иметь поле title — короткий интригующий заголовок для клипа:
    3–6 слов, заглавными буквами, без кавычек, без знаков "|", например "ОН НЕ ОЖИДАЛ ЭТОГО".
11. Если в этом фрагменте нет ничего стоящего, верни пустой список.

Верни СТРОГО JSON без пояснений:
{"moments":[{"start":12.5,"end":38.0,"category":"admiration","score":9,"reason":"...","title":"ОН НЕ ОЖИДАЛ ЭТОГО"}]}
"""


def _url(config: Config, path: str) -> str:
    return f"{config.ollama_url.rstrip('/')}{path}"


def status(config: Config) -> Tuple[bool, bool, str]:
    """Return (server_up, model_present, error_message)."""
    try:
        with urllib.request.urlopen(_url(config, "/api/tags"), timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, False, str(exc)

    wanted = config.ollama_model.lower()
    names = [str(model.get("name", "")).lower() for model in data.get("models", [])]
    present = any(name == wanted or name.startswith(wanted.split(":")[0] + ":") for name in names)
    return True, present, "" if present else f"модель {config.ollama_model} не найдена"


def is_available(config: Config) -> bool:
    server_up, model_present, _ = status(config)
    return server_up and model_present


def analyze_chunk(
    transcript: str,
    config: Config,
    *,
    offset: float,
    duration: float,
) -> List[Dict[str, Any]]:
    """Ask the local model for candidate moments in one transcript chunk.

    Raises on network/API errors so the caller can fall back to heuristics.
    """
    user_prompt = (
        f"Фрагмент стрима: таймкоды от {offset:.0f} до {min(offset + config.llm_chunk_seconds, duration):.0f} "
        f"секунд (всего {duration:.0f} c).\n"
        f"Максимум моментов: {config.max_clips}.\n\n"
        f"Расшифровка:\n{transcript}"
    )
    return _chat(SYSTEM_PROMPT, user_prompt, config)


RANK_PROMPT = """Ты — финальный редактор коротких видео.
Тебе дан список уже найденных интересных моментов в JSON.
Выбери лучшие и удали повторяющиеся/похожие. При необходимости чуть подправь
start и end так, чтобы момент начинался прямо с реакции. Сохрани category, score, reason
и title (короткий интригующий заголовок, 3–6 слов, заглавными буквами).
Верни СТРОГО JSON: {"moments":[{"start":...,"end":...,"category":"...","score":...,"reason":"...","title":"..."}]}
"""


def rank_moments(
    moments: List[Dict[str, Any]], config: Config
) -> List[Dict[str, Any]]:
    """Final editorial pass: deduplicate and refine boundaries globally."""
    if not moments:
        return []
    user_prompt = (
        f"Выбери не более {config.max_clips} лучших моментов.\n\n"
        + json.dumps({"moments": moments}, ensure_ascii=False)
    )
    return _chat(RANK_PROMPT, user_prompt, config, max_tokens=2000)


def _chat(
    system_prompt: str, user_prompt: str, config: Config, *, max_tokens: int = 1500
) -> List[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": config.ollama_model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.15, "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        _url(config, "/api/chat"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.llm_timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    return _parse(result.get("message", {}).get("content", ""))


def _parse(raw: str) -> List[Dict[str, Any]]:
    """Parse JSON even if the model wrapped it in prose or markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return []

    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []

    moments = obj.get("moments", []) if isinstance(obj, dict) else []
    if not isinstance(moments, list):
        return []
    return [m for m in moments if isinstance(m, dict)]
