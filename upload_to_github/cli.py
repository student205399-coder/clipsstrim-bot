"""Test the pipeline without a Telegram token.

Usage:
    python cli.py "https://youtube.com/watch?v=..."
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from clipper import process_url
from config import DEFAULT_CONFIG


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python cli.py "https://youtube.com/watch?v=..."')
    url = sys.argv[1]
    results = process_url(url, DEFAULT_CONFIG, print)
    if not results:
        print("Моменты не найдены.")
        return
    for item in results:
        path: Path = item["path"]
        print(f"{path}  [{item['start']}-{item['end']} c]  {item['category']}")


if __name__ == "__main__":
    main()
