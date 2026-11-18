"""Pipeline orchestration: download -> transcribe -> detect -> render."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import analyzer
import audio_cues
import downloader
import layout
import semantic
import subtitles
import transcriber
import visual_cues
from config import Config

ProgressFn = Callable[[str], None]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40] or "video"


def render_clip(
    video: Path,
    out_path: Path,
    moment: Dict[str, Any],
    words: List[Dict[str, Any]],
    box: Optional[Dict[str, int]],
    config: Config,
    work_dir: Path,
    index: int,
) -> Path:
    """Render one 9:16 clip, optionally with burned-in subtitles."""
    duration = max(0.1, float(moment["end"]) - float(moment["start"]))

    subtitle_name: Optional[str] = None
    if config.subtitles:
        ass_path = work_dir / f"subs_{index}.ass"
        subtitles.write_ass(ass_path, words, float(moment["start"]), float(moment["end"]))
        subtitle_name = ass_path.name

    title_name: Optional[str] = None
    title_text = str(moment.get("title", "")).strip()
    if title_text:
        title_path = work_dir / f"title_{index}.txt"
        title_path.write_text(title_text, encoding="utf-8")
        title_name = title_path.name

    filter_complex, output_label = layout.build_video_filter(
        config.layout,
        box,
        subtitle_name,
        title_file=title_name,
        duration=duration,
        title_color=config.title_color,
    )

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(moment["start"]),
        "-i",
        str(video),
        "-t",
        str(duration),
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{output_label}]",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    # cwd=work_dir lets the relative "ass=subs_x.ass" filter work on Windows.
    subprocess.run(command, cwd=str(work_dir), check=True)
    return out_path


def process_url(
    url: str,
    config: Config,
    progress: Optional[ProgressFn] = None,
    *,
    live: bool = False,
    tail_seconds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Full pipeline. Returns a list of {path, start, end, score, category}."""

    def say(message: str) -> None:
        if progress:
            progress(message)

    with tempfile.TemporaryDirectory(prefix="clipper_", dir=str(config.work_dir)) as tmp:
        work_dir = Path(tmp)

        say("⬇️ Скачиваю видео...")
        try:
            video = downloader.download(
                url, work_dir, tail_seconds=tail_seconds, live=live
            )
        except Exception as exc:
            if tail_seconds:
                say("⚠️ Не удалось скачать только хвост, беру целиком...")
                video = downloader.download(url, work_dir, live=live)
            else:
                raise exc

        duration = analyzer.probe_duration(video)
        say(f"📹 Видео {duration:.0f} c. Анализирую звук и речь...")

        # Webcam: manual override first, then automatic face detection.
        box = None
        if config.layout != "none":
            if config.webcam_box:
                width, height = analyzer.probe_size(video)
                fx, fy, fw, fh = config.webcam_box
                x = int(max(0, min(width - 2, fx * width)))
                y = int(max(0, min(height - 2, fy * height)))
                w = int(max(2, min(width - x, fw * width)))
                h = int(max(2, min(height - y, fh * height)))
                box = {
                    "x": x - x % 2,
                    "y": y - y % 2,
                    "w": w - w % 2,
                    "h": h - h % 2,
                    "frame_w": width,
                    "frame_h": height,
                }
                say("🎥 Вебка задана вручную (WEBCAM_BOX).")
            else:
                box = layout.detect_webcam_box(video, duration)
                if box:
                    say("🎥 Вебка найдена автоматически.")
                else:
                    say("🎥 Вебку не нашёл — кадр на весь экран.")

        say("🎧 Извлекаю звук и ищу реакции (крики, смех, всплески)...")
        wav_path = analyzer.extract_audio(video, work_dir / "audio.wav")
        features = audio_cues.extract_features(wav_path)
        cues = audio_cues.detect_cues(features)
        say(f"🎧 Найдено звуковых реакций: {len(cues)}.")

        transcript = transcriber.transcribe(video, config, audio_path=wav_path)
        say(f"🗣 Распознано слов: {len(transcript['words'])}.")

        # Preferred path: local LLM reads the transcript + audio markers.
        # None means it is unavailable -> heuristics.
        llm_result = None
        if config.llm_enabled:
            say("🧠 Ищу моменты через локальную модель (Ollama)...")
            llm_result = semantic.select_moments(
                transcript["words"], duration, config, cues=cues
            )

        if llm_result is not None:
            moments = llm_result
            engine = f"Ollama: {config.ollama_model} + аудио"
        else:
            if config.llm_enabled and not config.llm_fallback:
                say("❌ LLM недоступен, а откат выключен (LLM_FALLBACK=false).")
                return []
            say("⚠️ LLM недоступен — эвристика (громкость + словари + реакции).")
            peaks = analyzer.find_audio_peaks(features["rms"], features["times"])
            hits = analyzer.find_emotion_hits(transcript["words"])
            moments = analyzer.select_moments(
                peaks=peaks,
                hits=hits,
                rms=features["rms"],
                times=features["times"],
                duration=duration,
                config=config,
                cues=cues,
            )
            moments = analyzer.snap_starts(moments, cues, min_length=config.min_clip)
            engine = "эвристика + аудио"

        if not moments:
            say(f"🤷 Моменты не найдены ({engine}).")
            return []

        # Visual check runs ONLY on the chosen short clips, not the whole stream.
        if config.visual_enabled and visual_cues.available():
            say("👀 Проверяю мимику в отобранных моментах...")
            visual = visual_cues.analyze_candidates(video, box, moments, config)
            boosted = [cue for cue in visual if float(cue["score"]) >= config.visual_min_score]
            if boosted:
                moments = visual_cues.apply(moments, visual, config)
                say(f"👀 Визуальные реакции подтверждены: {len(boosted)}.")
            else:
                say("👀 Явных визуальных реакций не найдено.")

        say(f"✂️ Нарезаю {len(moments)} момент(ов) [{engine}]...")
        results: List[Dict[str, Any]] = []
        stem = _safe_name(video.stem)

        for index, moment in enumerate(moments):
            out_path = config.output_dir / f"clip_{stem}_{index:02d}.mp4"
            render_clip(
                video,
                out_path,
                moment,
                transcript["words"],
                box,
                config,
                work_dir,
                index,
            )
            results.append({**moment, "path": out_path})

        return results
