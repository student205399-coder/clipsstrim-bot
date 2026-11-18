"""Detect non-verbal reactions (screams, laughs, gasps) from audio only.

This runs locally, before the LLM, and gives the model explicit markers like
"[АУДИО в 45.2 c: высокий тон — крик/визг]".  The LLM can then use the spoken
context around that timestamp even when the reaction itself has no words.

Features are deliberately simple and cheap: loudness, zero-crossing rate
(proxy for pitch/noisiness), spectral centroid (brightness) and spectral flux
(sudden changes).  No model download, no GPU, works on any CPU.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

WINDOW_SECONDS = 0.5
_DECIMATION = 4  # 16 kHz -> 4 kHz is plenty for these features


def extract_features(wav_path: Path | str) -> Dict[str, np.ndarray]:
    """Compute per-window loudness / brightness / noisiness features."""
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    samples = samples[::_DECIMATION]
    rate = rate / _DECIMATION

    window = max(1, int(rate * WINDOW_SECONDS))
    count = len(samples) // window
    if count == 0:
        empty = np.array([0.0])
        return {"rms": empty, "zcr": empty, "centroid": empty, "flux": empty, "times": np.array([0.0])}

    frames = samples[: count * window].reshape(count, window)

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    signs = np.sign(frames)
    zcr = np.mean(np.abs(np.diff(signs, axis=1)), axis=1) / 2.0

    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    frequencies = np.fft.rfftfreq(window, d=1.0 / rate)
    centroid = (spectrum * frequencies).sum(axis=1) / (spectrum.sum(axis=1) + 1e-9)
    normalised = spectrum / (spectrum.sum(axis=1, keepdims=True) + 1e-9)
    flux = np.zeros(count, dtype=np.float32)
    flux[1:] = np.abs(np.diff(normalised, axis=0)).sum(axis=1)

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "flux": flux,
        "times": np.arange(count, dtype=np.float32) * WINDOW_SECONDS,
    }


def _z(values: np.ndarray) -> np.ndarray:
    return (values - float(values.mean())) / (float(values.std()) + 1e-9)


def detect_cues(
    features: Dict[str, np.ndarray],
    *,
    threshold: float = 1.6,
    min_gap: float = 2.0,
    max_cues: int = 40,
) -> List[Dict[str, Any]]:
    """Return notable audio events sorted by time."""
    rms = features["rms"]
    times = features["times"]
    if len(rms) < 4:
        return []

    loud = _z(rms)
    noisy = _z(features["zcr"])
    bright = _z(features["centroid"])
    sudden = _z(features["flux"])
    active = rms > float(rms.mean())

    score = np.maximum.reduce(
        [
            loud,
            np.where(active, noisy * 0.9, -np.inf),
            np.where(active, bright * 0.7, -np.inf),
            np.where(active, sudden * 0.8, -np.inf),
        ]
    )

    raw: List[Dict[str, Any]] = []
    for i in range(1, len(score) - 1):
        if score[i] < threshold or score[i] < score[i - 1] or score[i] <= score[i + 1]:
            continue
        if loud[i] >= noisy[i] and loud[i] >= sudden[i]:
            kind = "резкий всплеск громкости — крик/реакция"
        elif noisy[i] >= sudden[i]:
            kind = "высокий шум/тон — визг, смех или крик"
        else:
            kind = "резкая смена звука — удар/неожиданное событие"
        raw.append(
            {
                "time": float(times[i]),
                "kind": kind,
                "strength": float(min(1.0, score[i] / 4.0)),
                "score": float(score[i]),
            }
        )

    raw.sort(key=lambda cue: -cue["score"])
    selected: List[Dict[str, Any]] = []
    for cue in raw:
        if any(abs(cue["time"] - other["time"]) < min_gap for other in selected):
            continue
        selected.append(cue)
        if len(selected) >= max_cues:
            break

    return [
        {"time": round(c["time"], 2), "kind": c["kind"], "strength": round(c["strength"], 2)}
        for c in sorted(selected, key=lambda c: c["time"])
    ]


def format_cues(cues: List[Dict[str, Any]], limit: int = 30) -> str:
    """Render cue markers for the LLM prompt."""
    return "\n".join(
        f">>> [АУДИО в {cue['time']:.1f} c: {cue['kind']}]"
        for cue in cues[:limit]
    )
