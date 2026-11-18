"""Local speech-to-text with faster-whisper.

The model is downloaded once from HuggingFace and then cached locally, so after
the first run transcription is free, unlimited and works without any API key.

Audio is decoded to a 16 kHz mono WAV by ffmpeg and passed to Whisper as a
numpy array.  This avoids faster-whisper's PyAV code path, which is incompatible
with some PyAV versions (`av.open(..., metadata_errors=...)`).
"""

from __future__ import annotations

import wave
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from config import Config


@lru_cache(maxsize=2)
def _load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _read_wav(path: Path | str) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1:
            raise ValueError("Whisper requires 16 kHz mono WAV")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(
    video_path: Path | str, config: Config, *, audio_path: Optional[Path | str] = None
) -> Dict[str, Any]:
    """Return detected language and a flat list of timestamped words.

    Word-level timestamps are required for two things:
      * detecting emotion words at an exact moment in the stream;
      * rendering word-by-word "MrBeast style" subtitles.
    """
    model = _load_model(
        config.whisper_model, config.whisper_device, config.whisper_compute_type
    )
    audio = _read_wav(audio_path) if audio_path is not None else str(video_path)

    segments, info = model.transcribe(
        audio,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    words: List[Dict[str, Any]] = []
    for segment in segments:
        for word in segment.words or []:
            text = (word.word or "").strip()
            if not text:
                continue
            words.append(
                {
                    "word": text,
                    "start": float(word.start),
                    "end": float(word.end),
                    "prob": float(word.probability),
                }
            )

    return {"language": info.language, "words": words}
