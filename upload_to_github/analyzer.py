"""Find "interesting" moments locally.

Two signals are combined, both free and offline:

1. Loudness peaks in the audio (a scream, a laugh, a gasp = energy spike).
2. Emotion words/phrases from the Whisper transcript, e.g. "wow", "lets go",
   "fuck", "невероятно", "хахаха" (see ``emotion_words.py``).

Every clip is made to *start* on the reaction (a small pre-roll of ~0.7 s keeps
the beginning natural) and lasts between ``min_clip`` and ``max_clip`` seconds.
"""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from config import Config
from emotion_words import FLAT
import titles as title_tools

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)
_WINDOW_SECONDS = 0.5


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe helpers
# --------------------------------------------------------------------------- #
def probe_duration(video_path: Path | str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def probe_size(video_path: Path | str) -> Tuple[int, int]:
    """Return (width, height) of the first video stream."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = result.stdout.strip().split("x")[:2]
    return int(width), int(height)


def extract_audio(video_path: Path | str, out_path: Path | str) -> Path:
    """Decode audio to a mono 16 kHz WAV suitable for RMS analysis."""
    out_path = Path(out_path)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def compute_rms(wav_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rms values, timestamps) in ``_WINDOW_SECONDS`` steps."""
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    window = max(1, int(rate * _WINDOW_SECONDS))
    count = len(samples) // window
    if count == 0:
        return np.array([0.0]), np.array([0.0])

    trimmed = samples[: count * window].reshape(count, window)
    rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    times = np.arange(count, dtype=np.float32) * _WINDOW_SECONDS
    return rms, times


# --------------------------------------------------------------------------- #
# Signal detection
# --------------------------------------------------------------------------- #
def find_audio_peaks(
    rms: np.ndarray,
    times: np.ndarray,
    *,
    sensitivity: float = 1.2,
    min_gap: float = 3.0,
) -> List[Dict[str, float]]:
    """Local maxima of loudness that stand out from the average level."""
    if len(rms) < 3:
        return []

    mean = float(rms.mean())
    std = float(rms.std())
    threshold = mean + sensitivity * std

    raw: List[Tuple[float, float]] = []
    for i in range(1, len(rms) - 1):
        if rms[i] >= threshold and rms[i] >= rms[i - 1] and rms[i] > rms[i + 1]:
            raw.append((float(times[i]), float(rms[i])))

    raw.sort(key=lambda item: -item[1])
    selected: List[Tuple[float, float]] = []
    for time, value in raw:
        if all(abs(time - other) >= min_gap for other, _ in selected):
            selected.append((time, value))

    return [
        {
            "time": time,
            "score": min(3.0, max(0.0, (value - mean) / (std + 1e-9))),
        }
        for time, value in sorted(selected)
    ]


def find_emotion_hits(words: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find lexicon phrases in the transcript with their timestamps."""
    if not words:
        return []

    tokens: List[Dict[str, Any]] = []
    for word in words:
        text = _TOKEN_RE.sub("", str(word["word"]).lower())
        if text:
            tokens.append({"text": text, "start": float(word["start"])})

    hits: List[Dict[str, Any]] = []
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(t["text"] for t in tokens[i : i + n])
            entry = FLAT.get(phrase)
            if entry is None:
                continue
            category, weight = entry
            hits.append(
                {
                    "time": tokens[i]["start"],
                    "category": category,
                    "weight": float(weight),
                    "phrase": phrase,
                }
            )

    # Remove exact duplicates (the same phrase detected at the same word).
    unique: Dict[Tuple[float, str, str], Dict[str, Any]] = {}
    for hit in hits:
        key = (round(hit["time"], 2), hit["category"], hit["phrase"])
        unique[key] = hit
    return sorted(unique.values(), key=lambda h: h["time"])


def _loudness_at(
    time: float, rms: np.ndarray, mean: float, std: float, step: float
) -> float:
    index = int(round(time / step))
    index = max(0, min(len(rms) - 1, index))
    return max(0.0, (float(rms[index]) - mean) / (std + 1e-9))


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def select_moments(
    *,
    peaks: Sequence[Dict[str, float]],
    hits: Sequence[Dict[str, Any]],
    rms: np.ndarray,
    times: np.ndarray,
    duration: float,
    config: Config,
    cues: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Turn audio peaks + emotion hits into non-overlapping clip windows."""
    if duration <= 0:
        return []

    mean = float(rms.mean())
    std = float(rms.std())
    step = float(times[1] - times[0]) if len(times) > 1 else _WINDOW_SECONDS

    candidates: List[Dict[str, Any]] = []
    for hit in hits:
        audio = _loudness_at(hit["time"], rms, mean, std, step)
        candidates.append(
            {
                "time": float(hit["time"]),
                "score": hit["weight"] + 0.4 * min(3.0, audio),
                "category": hit["category"],
                "phrase": hit.get("phrase", ""),
            }
        )
    for peak in peaks:
        near_emotion = any(abs(peak["time"] - h["time"]) < 2.0 for h in hits)
        candidates.append(
            {
                "time": float(peak["time"]),
                "score": peak["score"] + (0.3 if near_emotion else 0.0),
                "category": "audio" if not near_emotion else "mixed",
                "phrase": "",
            }
        )
    for cue in cues:
        candidates.append(
            {
                "time": float(cue["time"]),
                "score": float(cue.get("strength", 0.5)) * 3.0,
                "category": "audio_reaction",
                "phrase": str(cue.get("kind", "")),
            }
        )

    # Merge candidates that are essentially the same moment.
    merged: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: -c["score"]):
        if any(abs(candidate["time"] - m["time"]) < 1.5 for m in merged):
            continue
        merged.append(candidate)

    # First give every candidate a window.  Stop a window shortly before the
    # next candidate so that one long clip cannot swallow all other moments.
    ordered = sorted(merged, key=lambda c: c["time"])
    windows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(ordered):
        start = max(0.0, candidate["time"] - 0.7)
        # Stronger reactions get longer clips.
        target = 30.0 + min(25.0, candidate["score"] * 6.0)
        target = max(config.min_clip, min(config.max_clip, target))

        next_time = ordered[index + 1]["time"] if index + 1 < len(ordered) else duration
        end = min(duration, start + target, next_time - 0.3)
        # Small epsilon avoids dropping a clip whose length is exactly
        # min_clip because of floating-point rounding.
        epsilon = 1e-6
        if end - start < config.min_clip - epsilon:
            end = min(duration, start + config.min_clip + 0.01)
        if end - start < config.min_clip - epsilon:
            continue

        windows.append({**candidate, "start": start, "end": end})

    # Then keep the strongest non-overlapping windows.
    moments: List[Dict[str, Any]] = []
    for window in sorted(windows, key=lambda w: -w["score"]):
        if len(moments) >= config.max_clips:
            break
        if any(
            _overlap((window["start"], window["end"]), (m["start"], m["end"]))
            > 0.25 * min(window["end"] - window["start"], m["end"] - m["start"])
            for m in moments
        ):
            continue

        moments.append(
            {
                "start": round(window["start"], 2),
                "end": round(window["end"], 2),
                "score": round(window["score"], 2),
                "category": window["category"],
                "phrase": window.get("phrase", ""),
                "title": title_tools.fallback(window["category"]),
            }
        )

    if moments:
        return sorted(moments, key=lambda m: m["start"])

    # Fallback: at least give the user one clip from the loudest part.
    if peaks:
        time = max(peaks, key=lambda p: p["score"])["time"]
        start = max(0.0, time - 0.7)
        end = min(duration, start + max(config.min_clip, min(config.max_clip, 30)))
        if end - start >= config.min_clip:
            return [
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "score": 0.0,
                    "category": "fallback",
                    "phrase": "",
                    "title": title_tools.fallback("fallback"),
                }
            ]

    if duration >= config.min_clip:
        return [
            {
                "start": 0.0,
                "end": round(min(duration, float(config.min_clip)), 2),
                "score": 0.0,
                "category": "fallback",
                "phrase": "",
                "title": title_tools.fallback("fallback"),
            }
        ]
    return []


def snap_starts(
    moments: Sequence[Dict[str, Any]],
    cues: Sequence[Dict[str, Any]],
    *,
    min_length: float = 15.0,
    radius: float = 2.5,
    preroll: float = 0.3,
) -> List[Dict[str, Any]]:
    """Move a clip's start onto the nearest real audio reaction.

    The LLM is good at choosing *which* moment, but its timestamp can be a
    little late.  If a detected audio spike sits just before the proposed
    start, we snap the beginning to that spike so the clip really starts with
    the reaction.
    """
    if not cues:
        return [dict(m) for m in moments]

    result: List[Dict[str, Any]] = []
    for moment in moments:
        start = float(moment["start"])
        end = float(moment["end"])
        nearby = [c for c in cues if start - radius <= float(c["time"]) <= start + 1.0]
        if nearby:
            strongest = max(nearby, key=lambda c: float(c.get("strength", 0)))
            candidate = max(0.0, float(strongest["time"]) - preroll)
            if candidate < start and end - candidate >= min_length:
                start = round(candidate, 2)
        result.append({**moment, "start": start})
    return result
