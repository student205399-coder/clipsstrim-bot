"""Generate ASS subtitles in the punchy "MrBeast" short-form style.

* big bold Impact font with a thick black outline;
* word-by-word karaoke: each word lights up yellow as it is spoken;
* emotion words get an extra colour;
* bottom-centre placement, tuned for a 1080x1920 (9:16) canvas.

Subtitles can be turned off completely with ``SUBTITLES=false`` or /subtitles.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from emotion_words import FLAT

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,92,&H00FFFFFF&,&H0000CCFF&,&H00000000&,&H64000000&,-1,0,0,0,100,100,0,0,1,7,3,2,40,40,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:
        secs += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")
    return re.sub(r"\s+", " ", text).strip()


def _is_emotion(word: str) -> bool:
    return _TOKEN_RE.sub("", word.lower()) in FLAT


def _build_lines(
    words: Sequence[Dict[str, Any]], clip_start: float, clip_end: float
) -> List[List[Dict[str, Any]]]:
    """Group words into short subtitle lines."""
    relevant = [
        w
        for w in words
        if float(w["end"]) >= clip_start and float(w["start"]) <= clip_end
    ]
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for word in relevant:
        start = float(word["start"])
        if current:
            gap = start - float(current[-1]["end"])
            span = start - float(current[0]["start"])
            if gap > 0.5 or span > 3.5 or len(current) >= 8:
                lines.append(current)
                current = []
        current.append(word)

    if current:
        lines.append(current)
    return lines


def write_ass(
    out_path: Path | str,
    words: Sequence[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> Path:
    """Write an ASS subtitle file whose timestamps are relative to the clip."""
    out_path = Path(out_path)
    content = [_ASS_HEADER]

    for line in _build_lines(words, clip_start, clip_end):
        line_start = max(clip_start, float(line[0]["start"]))
        line_end = min(clip_end, float(line[-1]["end"]))
        if line_end <= line_start:
            continue

        pieces: List[str] = []
        for word in line:
            word_start = max(clip_start, float(word["start"]))
            word_end = min(clip_end, float(word["end"]))
            duration_cs = max(1, int(round((word_end - word_start) * 100)))
            colour = "&H0000CCFF&" if _is_emotion(str(word["word"])) else "&H00FFFFFF&"
            pieces.append(
                r"{\1c" + colour + r"}{\kf" + str(duration_cs) + "}" + _escape(str(word["word"])) + " "
            )

        text = r"{\fad(120,120)}" + "".join(pieces)
        content.append(
            "Dialogue: 0,"
            f"{_format_time(line_start - clip_start)},"
            f"{_format_time(line_end - clip_start)},"
            "Default,,0,0,0,," + text + "\n"
        )

    out_path.write_text("".join(content), encoding="utf-8")
    return out_path
