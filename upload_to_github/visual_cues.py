"""Lightweight visual reaction analysis for the *already selected* clips.

We do NOT scan the whole stream.  For each 15–60 s candidate we sample a handful
of frames, look at the face (inside the detected webcam box when available) and
detect smiles / open mouths / sharp leaning motions with OpenCV Haar cascades.
This catches silent visual reactions that speech and audio analysis cannot see,
without downloading a heavy video model.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the user's environment
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

import numpy as np

from cascade_paths import find as find_cascade
from config import Config

logger = logging.getLogger("clipper_bot.visual")


def available() -> bool:
    return CV2_AVAILABLE


@lru_cache(maxsize=1)
def _cascades() -> Tuple[List[Any], Dict[str, Any]]:
    """Load face + smile/eye/mouth cascades once."""
    if not CV2_AVAILABLE:
        return [], {}

    faces: List[Any] = []
    for name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    ):
        path = find_cascade(name)
        if path is not None:
            classifier = cv2.CascadeClassifier(str(path))
            if not classifier.empty():
                faces.append(classifier)

    extras: Dict[str, Any] = {}
    for key, name in (
        ("smile", "haarcascade_smile.xml"),
        ("eye", "haarcascade_eye.xml"),
    ):
        path = find_cascade(name)
        if path is not None:
            classifier = cv2.CascadeClassifier(str(path))
            if not classifier.empty():
                extras[key] = classifier
    return faces, extras


def analyze_candidates(
    video: Path | str,
    box: Optional[Dict[str, int]],
    moments: Sequence[Dict[str, Any]],
    config: Config,
) -> List[Dict[str, Any]]:
    """Return one visual-cue record per candidate moment."""
    if not CV2_AVAILABLE or not moments:
        return []
    faces, extras = _cascades()
    if not faces:
        return []

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []

    sample_count = max(3, int(config.visual_samples))
    results: List[Dict[str, Any]] = []

    for moment in moments:
        start = float(moment["start"])
        end = float(moment["end"])
        timestamps = np.linspace(start + 0.4, max(start + 0.4, end - 0.4), sample_count)

        smiles = 0
        mouths = 0
        faces_seen = 0
        face_sizes: List[float] = []
        best: Tuple[float, int] = (start, 0)

        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue

            if box:
                bx, by = int(box["x"]), int(box["y"])
                bw, bh = int(box["w"]), int(box["h"])
                crop = frame[by : by + bh, bx : bx + bw]
            else:
                crop = frame
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            detected: List[Any] = []
            for classifier in faces:
                found = classifier.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(found):
                    detected.extend(found)
            if not detected:
                continue

            x, y, w, h = max(detected, key=lambda rect: rect[2] * rect[3])
            faces_seen += 1
            face_sizes.append(float(h))
            face_roi = gray[y : y + h, x : x + w]
            hits = 0

            smile_cascade = extras.get("smile")
            if smile_cascade is not None:
                found_smiles = smile_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.7,
                    minNeighbors=18,
                    minSize=(max(10, int(w * 0.3)), max(6, int(h * 0.15))),
                )
                if len(found_smiles):
                    smiles += 1
                    hits += 1

            mouth_cascade = extras.get("mouth")
            if mouth_cascade is not None:
                found_mouths = mouth_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=1.7,
                    minNeighbors=12,
                    minSize=(max(10, int(w * 0.25)), max(6, int(h * 0.12))),
                )
                if len(found_mouths):
                    mouths += 1
                    hits += 1

            if hits > best[1]:
                best = (float(timestamp), hits)

        capture.release()

        score = 0.0
        labels: List[str] = []
        threshold = max(2, sample_count // 4)

        if smiles >= threshold:
            score += 0.45
            labels.append("улыбка/смех на лице")
        if mouths >= threshold:
            score += 0.55
            labels.append("открытый рот — шок/крик")
        if len(face_sizes) >= 2 and max(face_sizes) / max(1.0, min(face_sizes)) > 1.3:
            score += 0.25
            labels.append("резкое движение/наклон к камере")

        results.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "score": round(min(1.0, score), 2),
                "face_frames": faces_seen,
                "labels": labels,
                "reaction_time": round(best[0], 2) if best[1] > 0 else None,
            }
        )

    return results


def apply(
    moments: Sequence[Dict[str, Any]],
    visual: Sequence[Dict[str, Any]],
    config: Config,
) -> List[Dict[str, Any]]:
    """Boost visually confirmed moments and snap their start to the reaction."""
    lookup = {(round(c["start"], 2), round(c["end"], 2)): c for c in visual}
    updated: List[Dict[str, Any]] = []

    for moment in moments:
        cue = lookup.get((round(float(moment["start"]), 2), round(float(moment["end"]), 2)))
        if cue is None or float(cue["score"]) < config.visual_min_score:
            updated.append(dict(moment))
            continue

        result = dict(moment)
        result["score"] = round(float(moment.get("score", 0)) + float(cue["score"]) * 2.0, 2)
        reason = str(moment.get("phrase", ""))
        labels = ", ".join(cue["labels"])
        result["phrase"] = f"{reason}; {labels}" if reason else labels

        reaction = cue.get("reaction_time")
        if reaction is not None:
            time = float(reaction)
            if float(moment["start"]) - 2.5 <= time <= float(moment["start"]) + 1.0:
                new_start = max(0.0, time - 0.3)
                if float(moment["end"]) - new_start >= config.min_clip:
                    result["start"] = round(new_start, 2)

        updated.append(result)

    return updated
