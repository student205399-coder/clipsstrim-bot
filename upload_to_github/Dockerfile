FROM python:3.12-slim

# ffmpeg with libass (for ASS subtitles), freetype/fribidi (for drawtext title)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libass9 libfreetype6 libfribidi0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
