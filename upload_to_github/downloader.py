"""Download source videos with yt-dlp (YouTube, and most other sites)."""

from __future__ import annotations

import os
from pathlib import Path

from yt_dlp import YoutubeDL

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".ts"}


def _proxy() -> str:
    """Use the same local proxy as Telegram when YouTube is unreachable."""
    return os.getenv("DOWNLOAD_PROXY") or os.getenv("TELEGRAM_PROXY") or ""


def download(
    url: str,
    out_dir: Path | str,
    *,
    tail_seconds: int | None = None,
    live: bool = False,
) -> Path:
    """Download *url* into *out_dir* and return the resulting video file.

    Args:
        url: YouTube / Twitch / any URL supported by yt-dlp.
        out_dir: directory for temporary files.
        tail_seconds: if set, try to keep only the last N seconds.  Useful for
            monitoring a live stream without re-downloading hours of video.
            yt-dlp implements this with ``--download-sections``.
        live: enable yt-dlp's ``live_from_start`` for live streams.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    proxy = _proxy()
    if proxy:
        options["proxy"] = proxy
    if live:
        options["live_from_start"] = True
    if tail_seconds and tail_seconds > 0:
        options["download_sections"] = [f"*-{int(tail_seconds)}"]

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = str(info.get("id", "video"))

    # yt-dlp may produce id.mp4 / id.webm / id.f137.mp4 etc.  Pick the newest
    # video file that belongs to this download.
    candidates = [
        p
        for p in out_dir.glob(f"{video_id}*")
        if p.suffix.lower() in _VIDEO_SUFFIXES and p.is_file()
    ]
    if not candidates:
        raise RuntimeError(
            f"yt-dlp did not produce a video file for {url!r} in {out_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_live(url: str) -> bool:
    """Best-effort check whether a URL points to a live stream."""
    options = {"quiet": True, "no_warnings": True, "skip_download": True}
    proxy = _proxy()
    if proxy:
        options["proxy"] = proxy
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        return bool(info.get("live_status") in {"live", "is_live"})
    except Exception:
        return False
