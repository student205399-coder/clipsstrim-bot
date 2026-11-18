"""Locate the embedded webcam and build 9:16 ffmpeg layouts.

Stream recordings usually have the facecam burned into one corner.  We sample a
few frames, run OpenCV's Haar face detector, and infer a square webcam region
around the most common face position.  This is a heuristic: layered/ANIMATED
webcams or unusual overlays may need manual tweaking of ``_expand_box``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from cascade_paths import find as find_cascade
import titles as title_tools


def detect_webcam_box(video_path: Path | str, duration: float) -> Optional[Dict[str, int]]:
    """Return {x, y, w, h, frame_w, frame_h} for the facecam, or None."""
    cascades = []
    for name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ):
        path = find_cascade(name)
        if path is None:
            continue
        classifier = cv2.CascadeClassifier(str(path))
        if not classifier.empty():
            cascades.append(classifier)
    if not cascades:
        return None

    capture = cv2.VideoCapture(str(video_path))
    frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        capture.release()
        return None

    if duration > 4:
        sample_times = np.linspace(1.0, max(2.0, duration - 1.0), 18)
    else:
        sample_times = np.linspace(0.0, max(0.5, duration), 5)

    min_face = max(30, int(frame_w * 0.04))
    faces: list[Tuple[int, int, int, int]] = []

    for timestamp in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = []
        for classifier in cascades:
            found = classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face, min_face),
            )
            if len(found):
                detected.extend(found)
        if detected:
            x, y, w, h = max(detected, key=lambda rect: rect[2] * rect[3])
            faces.append((int(x), int(y), int(w), int(h)))

    capture.release()
    if not faces:
        return None

    x, y, w, h = (int(np.median(values)) for values in zip(*faces))

    # Expand the detected face into a square webcam overlay.
    size = int(np.clip(w * 3.2, frame_w * 0.15, frame_w * 0.42))
    size = max(2, min(size, frame_h))
    centre_x, centre_y = x + w / 2.0, y + h / 2.0
    bx = int(np.clip(centre_x - size / 2.0, 0, frame_w - size))
    by = int(np.clip(centre_y - size / 2.0, 0, frame_h - size))

    # x264/yuv420p prefers even crop dimensions.
    if size % 2:
        size -= 1
    if bx % 2:
        bx = min(bx + 1, frame_w - size)
    if by % 2:
        by = min(by + 1, frame_h - size)

    return {
        "x": bx,
        "y": by,
        "w": size,
        "h": size,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def _fill_crop(source: str, width: int, height: int) -> str:
    """Crop/scale a source so it completely fills width x height."""
    # A bare "[0:v]" must be followed directly by the first filter; a chain
    # like "[0:v]crop=w:h:x:y" already ends with a filter and needs a comma.
    separator = "" if source.endswith("]") else ","
    return (
        f"{source}{separator}scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )


_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/dejavu-sans-bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
)


@lru_cache(maxsize=1)
def _font_path() -> Optional[str]:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _drawtext(
    font: str,
    title_file: str,
    *,
    boxed: bool = False,
    y: str = "(h-text_h)/2",
    color: str = "white",
) -> str:
    """Build a drawtext filter that reads a UTF-8 text file (Cyrillic-safe)."""
    # ffmpeg's filtergraph parser needs a doubled backslash before ':' on
    # Windows drive letters.
    escaped_font = font.replace(":", r"\\:")
    parts = [
        f"fontfile={escaped_font}",
        f"textfile={title_file}",
        "expansion=none",
        "fontsize=54",
        f"fontcolor={color}",
        "x=(w-text_w)/2",
        f"y={y}",
        "shadowx=2",
        "shadowy=2",
    ]
    if boxed:
        parts.extend(["box=1", "boxcolor=black@0.7", "boxborderw=16"])
    return "drawtext=" + ":".join(parts)


def build_video_filter(
    mode: str,
    box: Optional[Dict[str, int]],
    subtitle_name: Optional[str],
    *,
    title_file: Optional[str] = None,
    duration: float = 0.0,
    title_color: str = "white",
) -> Tuple[str, str]:
    """Build a filter_complex string and the final output label.

    Args:
        mode: ``none``, ``webcam_full`` or ``top_webcam_bottom_screen``.
        box: webcam rectangle from :func:`detect_webcam_box`.
        subtitle_name: basename of the ASS file in the ffmpeg working dir.
        title_file: basename of a UTF-8 file with the intriguing title.
        duration: clip length, used to bound the generated title band.

    Returns:
        (filter_complex, output_label)
    """
    mode = (mode or "none").lower()
    font = _font_path() if title_file else None
    color = title_tools.color(title_color)
    output_label = "out"
    title_in_band = False

    if mode == "webcam_full" and box:
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        chain = _fill_crop(f"[0:v]crop={bw}:{bh}:{bx}:{by}", 1080, 1920) + "[out]"
    elif mode == "top_webcam_bottom_screen" and box:
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        frame_w, frame_h = box["frame_w"], box["frame_h"]

        # The "screen" is the part of the frame that is not the webcam.
        if bx + bw / 2.0 < frame_w / 2.0:
            sx, sy = bx + bw, 0
            sw, sh = frame_w - sx, frame_h
        else:
            sx, sy = 0, 0
            sw, sh = bx, frame_h
        if sw < frame_w * 0.3 or sh < frame_h * 0.3:
            sx, sy, sw, sh = 0, 0, frame_w, frame_h

        if title_file and font:
            # webcam 560 + title band 180 + gameplay 1180 = 1920
            top = _fill_crop(f"[0:v]crop={bw}:{bh}:{bx}:{by}", 1080, 560) + "[top]"
            band = (
                f"color=c=0x141414:s=1080x180:d={max(0.1, duration):.2f},format=yuv420p,"
                f"{_drawtext(font, title_file, color=color)}[band]"
            )
            bottom = _fill_crop(f"[0:v]crop={sw}:{sh}:{sx}:{sy}", 1080, 1180) + "[bottom]"
            chain = f"{top};{band};{bottom};[top][band][bottom]vstack=inputs=3[out]"
            title_in_band = True
        else:
            top = _fill_crop(f"[0:v]crop={bw}:{bh}:{bx}:{by}", 1080, 620) + "[top]"
            bottom = _fill_crop(f"[0:v]crop={sw}:{sh}:{sx}:{sy}", 1080, 1300) + "[bottom]"
            chain = top + ";" + bottom + ";[top][bottom]vstack=inputs=2[out]"
    else:
        # "none" mode, or a webcam mode where no face was found.
        chain = _fill_crop("[0:v]", 1080, 1920) + "[out]"

    if title_file and font and not title_in_band:
        chain += (
            f";[out]{_drawtext(font, title_file, boxed=True, y='90', color=color)}[titled]"
        )
        output_label = "titled"

    if subtitle_name:
        # Relative filename: ffmpeg runs with cwd set to the temp directory.
        chain += f";[{output_label}]ass={subtitle_name}[subbed]"
        return chain, "subbed"

    return chain, output_label
