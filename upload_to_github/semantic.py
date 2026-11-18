"""Semantic moment detection: let a local LLM read the whole transcript.

The model understands *meaning* — irony, drama, surprise, context — which a
keyword list cannot.  Audio cues (screams/laughs without words) are injected as
explicit markers so the model can use them too.  Long transcripts are split
into chunks; results are validated, ranked and de-duplicated.  If Ollama is
unavailable or every call fails, the caller falls back to ``analyzer.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import analyzer
import llm
import titles as title_tools
from config import Config

logger = logging.getLogger("clipper_bot.semantic")


def build_lines(words: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group timestamped words into short readable transcript lines."""
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for word in words:
        start = float(word["start"])
        if current:
            gap = start - float(current[-1]["end"])
            if gap > 1.5 or len(current) >= 14:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    lines: List[Dict[str, Any]] = []
    for group in groups:
        text = " ".join(str(w["word"]).strip() for w in group)
        text = " ".join(text.split())
        if not text:
            continue
        lines.append(
            {
                "start": float(group[0]["start"]),
                "end": float(group[-1]["end"]),
                "text": text[:240],
            }
        )
    return lines


def chunk_lines(
    lines: Sequence[Dict[str, Any]], window_seconds: int, max_chars: int = 12000
) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """Split lines into chunks of at most ``window_seconds`` / ``max_chars``."""
    if not lines:
        return []

    chunks: List[Tuple[float, List[Dict[str, Any]]]] = []
    current: List[Dict[str, Any]] = []
    start = float(lines[0]["start"])
    chars = 0

    for line in lines:
        if current and (
            float(line["end"]) - start > window_seconds
            or chars + len(line["text"]) > max_chars
        ):
            chunks.append((start, current))
            current = []
            start = float(line["start"])
            chars = 0
        current.append(line)
        chars += len(line["text"]) + 12

    if current:
        chunks.append((start, current))
    return chunks


def _render_chunk(
    lines: Sequence[Dict[str, Any]], cues: Sequence[Dict[str, Any]]
) -> str:
    """Merge speech and audio-cue markers into one chronological transcript."""
    entries: List[Tuple[float, str]] = [
        (float(line["start"]), f"[{line['start']:.1f}] {line['text']}")
        for line in lines
    ]
    entries.extend(
        (
            float(cue["time"]),
            f">>> [АУДИО в {cue['time']:.1f} c: {cue['kind']}]",
        )
        for cue in cues
    )
    entries.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in entries[:80])


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _validate(
    raw_moments: Sequence[Dict[str, Any]], duration: float, config: Config
) -> List[Dict[str, Any]]:
    moments: List[Dict[str, Any]] = []
    for item in raw_moments:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not (0 <= start < end <= duration):
            continue

        # Normalise length while keeping the reaction at the very beginning.
        if end - start < config.min_clip:
            end = min(duration, start + config.min_clip)
        if end - start > config.max_clip:
            end = start + config.max_clip
        if end - start < config.min_clip:
            continue

        try:
            score = float(item.get("score", 5))
        except (TypeError, ValueError):
            score = 5.0
        category = str(item.get("category", "shock"))[:32]
        reason = str(item.get("reason", ""))[:200]
        title = title_tools.clean(item.get("title"), category)

        moments.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "score": round(score, 2),
                "category": category,
                "phrase": reason,
                "title": title,
            }
        )
    return moments


def _dedupe(candidates: Sequence[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: -c["score"]):
        if len(selected) >= config.max_clips:
            break
        if any(
            _overlap((candidate["start"], candidate["end"]), (m["start"], m["end"]))
            > 0.25 * min(candidate["end"] - candidate["start"], m["end"] - m["start"])
            for m in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda m: m["start"])


def select_moments(
    words: Sequence[Dict[str, Any]],
    duration: float,
    config: Config,
    cues: Sequence[Dict[str, Any]] = (),
) -> Optional[List[Dict[str, Any]]]:
    """Return LLM-selected moments, or None when the LLM cannot be used.

    None means "fall back to heuristics".  An empty list is a valid answer: the
    LLM was available but found nothing worth clipping.
    """
    if not config.llm_enabled or not llm.is_available(config):
        return None
    if len(words) < 20:  # too little speech — let the audio heuristics decide
        return None

    lines = build_lines(words)
    if not lines:
        return None

    chunks = chunk_lines(lines, config.llm_chunk_seconds)[: config.llm_max_chunks]
    raw: List[Dict[str, Any]] = []
    successes = 0

    for offset, chunk in chunks:
        chunk_cues = [
            cue for cue in cues if offset <= float(cue["time"]) <= offset + config.llm_chunk_seconds
        ]
        try:
            raw.extend(
                llm.analyze_chunk(
                    _render_chunk(chunk, chunk_cues),
                    config,
                    offset=offset,
                    duration=duration,
                )
            )
            successes += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama chunk analysis failed: %s", exc)

    if successes == 0:
        return None

    candidates = _validate(raw, duration, config)
    selected = _dedupe(candidates, config)

    # Final global editorial pass: remove similar clips from different chunks
    # and sharpen boundaries.  If it fails, keep the original selection.
    if config.llm_global_rank and len(selected) > 1:
        try:
            refined = llm.rank_moments(
                [
                    {key: value for key, value in m.items() if key != "phrase"}
                    for m in selected
                ],
                config,
            )
            validated = _validate(refined, duration, config)
            if validated:
                selected = _dedupe(validated, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama global ranking failed: %s", exc)

    return analyzer.snap_starts(selected, cues, min_length=config.min_clip)
