FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libass9 libfreetype6 libfribidi0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY upload_to_github/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY upload_to_github/ .

CMD ["python", "bot.py"]
