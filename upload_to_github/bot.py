"""Telegram bot entry point.

Commands:
  /start                 - help
  /clip <url>            - make clips from a YouTube (or other) link
  /subtitles on|off      - toggle MrBeast-style subtitles
  /titlecolor <color>    - white | green | yellow | red | blue title on the dark band
  /layout <mode>         - none | webcam_full | top_webcam_bottom_screen
  /auto <url> [minutes]  - monitor a live stream and send best moments every N min
  /stopauto              - stop monitoring in this chat
  /settings              - current settings
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load .env before importing config: DEFAULT_CONFIG is created at import time.
load_dotenv()

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from clipper import process_url
from config import DEFAULT_CONFIG, Config
import llm
import titles as title_tools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("clipper_bot")

# Per-chat state (single-process bot).
USER_CONFIGS: Dict[int, Config] = {}
MONITOR_TASKS: Dict[int, asyncio.Task] = {}


class Progress:
    """Sync progress callback that can be called from a worker thread."""

    def __init__(self, bot, chat_id: int, loop: asyncio.AbstractEventLoop) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.loop = loop

    def send(self, text: str) -> None:
        asyncio.run_coroutine_threadsafe(
            self.bot.send_message(chat_id=self.chat_id, text=text), self.loop
        ).result()


def _config_for(update: Update) -> Config:
    chat_id = update.effective_chat.id
    if chat_id not in USER_CONFIGS:
        USER_CONFIGS[chat_id] = DEFAULT_CONFIG.copy()
    return USER_CONFIGS[chat_id]


async def _send_results(bot, chat_id: int, results, subtitle_note: str = "") -> None:
    for item in results:
        path: Path = item["path"]
        caption = (
            f"🔥 Момент {item['start']:.0f}–{item['end']:.0f} c "
            f"({item['category']}, score {item['score']})"
        )
        reason = str(item.get("phrase", ""))
        if reason:
            caption += f"\n💡 {reason[:160]}"
        if subtitle_note:
            caption += f"\n{subtitle_note}"
        with path.open("rb") as video_file:
            await bot.send_video(
                chat_id=chat_id,
                video=video_file,
                caption=caption,
                supports_streaming=True,
                read_timeout=120,
                write_timeout=120,
            )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я делаю короткие вертикальные нарезки (9:16) из YouTube-видео.\n\n"
        "Команды:\n"
        "/clip <ссылка> — скачать, найти эмоциональные моменты и прислать нарезки\n"
        "/subtitles on|off — субтитры как у Мистера Биста\n"
        "/titlecolor white|green|yellow|red|blue — цвет заголовка\n"
        "/layout none|webcam_full|top_webcam_bottom_screen — раскладка\n"
        "/ai on|off — локальный LLM (Ollama) для поиска моментов по смыслу\n"
        "/webcam x,y,w,h|auto — прямоугольник вебки в долях кадра\n"
        "/auto <ссылка> [минуты] — следить за эфиром и кидать лучший момент\n"
        "/stopauto — остановить слежение\n"
        "/settings — текущие настройки\n\n"
        "Всё считается локально: Whisper + ffmpeg, без платных API."
    )


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = _config_for(update)
    if cfg.llm_enabled:
        server_up, model_present, error = await asyncio.to_thread(llm.status, cfg)
        if server_up and model_present:
            brain = f"Ollama ({cfg.ollama_model})"
        elif server_up:
            brain = f"Ollama: {error}"
        else:
            brain = f"Ollama недоступен: {error}"
    else:
        brain = "выключен (эвристика)"

    await update.message.reply_text(
        f"Мозг: {brain}\n"
        f"Субтитры: {'вкл' if cfg.subtitles else 'выкл'}\n"
        f"Раскладка: {cfg.layout}\n"
        f"Цвет заголовка: {cfg.title_color}\n"
        f"Вебка: {cfg.webcam_box or 'auto (по лицу)'}\n"
        f"Длительность: {cfg.min_clip}–{cfg.max_clip} c\n"
        f"Моментов за видео: до {cfg.max_clips}\n"
        f"Интервал мониторинга: {cfg.monitor_interval_min} мин\n"
        f"Модель Whisper: {cfg.whisper_model} ({cfg.whisper_device})"
    )


async def clip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    url = " ".join(context.args or []).strip()
    if not url:
        await update.message.reply_text("Usage: /clip https://youtube.com/watch?v=...")
        return

    cfg = _config_for(update)
    chat_id = update.effective_chat.id
    progress = Progress(context.application.bot, chat_id, asyncio.get_running_loop())

    status = await update.message.reply_text("🚀 Начинаю...")
    try:
        results = await asyncio.to_thread(process_url, url, cfg, progress.send)
    except Exception as exc:  # noqa: BLE001 - show the real error to the user
        logger.exception("Clip failed")
        await status.edit_text(f"❌ Ошибка: {exc}")
        return

    if not results:
        await status.edit_text("🤷 Не нашёл подходящих моментов.")
        return

    await status.edit_text(f"✅ Готово, нарезок: {len(results)}. Отправляю...")
    await _send_results(context.application.bot, chat_id, results)


async def subtitles_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    cfg = _config_for(update)
    if not args:
        await update.message.reply_text("Usage: /subtitles on|off")
        return
    value = args[0].lower()
    if value not in {"on", "off", "вкл", "выкл"}:
        await update.message.reply_text("Usage: /subtitles on|off")
        return
    cfg.subtitles = value in {"on", "вкл"}
    await update.message.reply_text(
        f"Субтитры: {'включены' if cfg.subtitles else 'выключены'}"
    )


async def titlecolor_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = _config_for(update)
    args = context.args or []
    if not args:
        await update.message.reply_text(
            f"Цвет заголовка: {cfg.title_color}. "
            "Доступно: white / green / yellow / red / blue (чёрная полоса остаётся)."
        )
        return

    key = args[0].lower()
    if key not in title_tools.COLOR_ALIASES:
        await update.message.reply_text(
            "Цвета: white, green, yellow, red, blue"
        )
        return
    cfg.title_color = title_tools.COLOR_ALIASES[key]
    await update.message.reply_text(f"Цвет заголовка: {cfg.title_color}")


async def layout_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    cfg = _config_for(update)
    allowed = {"none", "webcam_full", "top_webcam_bottom_screen"}
    if not args or args[0] not in allowed:
        await update.message.reply_text(
            "Usage: /layout none|webcam_full|top_webcam_bottom_screen"
        )
        return
    cfg.layout = args[0]
    await update.message.reply_text(f"Раскладка: {cfg.layout}")


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = _config_for(update)
    args = context.args or []
    if not args:
        server_up, model_present, error = await asyncio.to_thread(llm.status, cfg)
        state = "вкл" if cfg.llm_enabled else "выкл"
        detail = "OK" if model_present else (error or "сервер не отвечает")
        await update.message.reply_text(
            f"Локальный LLM: {state}. Ollama: {detail}.\n"
            "Usage: /ai on|off"
        )
        return

    value = args[0].lower()
    if value not in {"on", "off", "вкл", "выкл"}:
        await update.message.reply_text("Usage: /ai on|off")
        return
    cfg.llm_enabled = value in {"on", "вкл"}
    await update.message.reply_text(
        f"Поиск моментов через LLM: {'включён' if cfg.llm_enabled else 'выключен'}"
    )


async def webcam_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg = _config_for(update)
    args = context.args or []
    if not args:
        await update.message.reply_text(
            f"Вебка: {cfg.webcam_box or 'auto (по лицу)'}\n"
            "Usage: /webcam 0.02,0.05,0.26,0.29  или  /webcam auto"
        )
        return

    if args[0].lower() == "auto":
        cfg.webcam_box = None
        await update.message.reply_text("Вебка: авто-определение по лицу.")
        return

    try:
        values = tuple(float(part) for part in args[0].replace(";", ",").split(","))
        if (
            len(values) != 4
            or any(v < 0 or v > 1 for v in values)
            or values[2] <= 0
            or values[3] <= 0
            or values[0] + values[2] > 1.001
            or values[1] + values[3] > 1.001
        ):
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "Usage: /webcam x,y,w,h (доли кадра 0..1) или /webcam auto"
        )
        return

    cfg.webcam_box = values  # type: ignore[assignment]
    await update.message.reply_text(f"Вебка задана: {values}")


async def _monitor_loop(
    bot,
    chat_id: int,
    url: str,
    cfg: Config,
    interval_minutes: int,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Periodically re-check the stream and send 1–2 best moments."""
    monitor_cfg = cfg.copy(max_clips=cfg.monitor_max_clips)
    while True:
        try:
            progress = Progress(bot, chat_id, loop)
            job = partial(
                process_url,
                url,
                monitor_cfg,
                progress.send,
                live=True,
                tail_seconds=cfg.monitor_tail_seconds,
            )
            results = await asyncio.to_thread(job)
            if results:
                await _send_results(bot, chat_id, results, "👀 Автомониторинг")
            else:
                await bot.send_message(chat_id=chat_id, text="👀 Новых моментов не нашёл.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Monitor iteration failed")
            await bot.send_message(
                chat_id=chat_id, text=f"⚠️ Ошибка мониторинга: {exc}"
            )
        await asyncio.sleep(max(1, interval_minutes) * 60)


async def auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /auto <ссылка> [минуты]")
        return

    url = args[0]
    try:
        interval = int(args[1]) if len(args) > 1 else None
    except ValueError:
        interval = None

    chat_id = update.effective_chat.id
    cfg = _config_for(update)
    interval = interval or cfg.monitor_interval_min

    old = MONITOR_TASKS.pop(chat_id, None)
    if old and not old.done():
        old.cancel()

    task = asyncio.create_task(
        _monitor_loop(
            context.application.bot,
            chat_id,
            url,
            cfg,
            interval,
            asyncio.get_running_loop(),
        )
    )
    MONITOR_TASKS[chat_id] = task
    await update.message.reply_text(
        f"👀 Слежу за эфиром каждые {interval} мин. "
        "Первая проверка — сейчас. /stopauto чтобы остановить."
    )


async def stopauto_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    task = MONITOR_TASKS.pop(chat_id, None)
    if task and not task.done():
        task.cancel()
        await update.message.reply_text("🛑 Слежение остановлено.")
    else:
        await update.message.reply_text("Слежение не запущено.")


def build_application(token: str) -> Application:
    builder = Application.builder().token(token)
    proxy = os.getenv("TELEGRAM_PROXY") or DEFAULT_CONFIG.telegram_proxy
    if proxy:
        # Needed when api.telegram.org is unreachable directly.
        builder = builder.proxy(proxy).get_updates_proxy(proxy)
    application = builder.build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("settings", settings))
    application.add_handler(CommandHandler("clip", clip_command))
    application.add_handler(CommandHandler("subtitles", subtitles_command))
    application.add_handler(CommandHandler("titlecolor", titlecolor_command))
    application.add_handler(CommandHandler("layout", layout_command))
    application.add_handler(CommandHandler("ai", ai_command))
    application.add_handler(CommandHandler("webcam", webcam_command))
    application.add_handler(CommandHandler("auto", auto_command))
    application.add_handler(CommandHandler("stopauto", stopauto_command))
    return application


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or DEFAULT_CONFIG.telegram_token
    if not token:
        raise SystemExit(
            "Укажи TELEGRAM_BOT_TOKEN в переменных окружения или в config.py"
        )
    build_application(token).run_polling()


if __name__ == "__main__":
    main()
