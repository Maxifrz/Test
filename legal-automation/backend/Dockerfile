FROM python:3.12-slim AS base

# ffmpeg: Audio-Normalisierung vor der Transkription
# tesseract-ocr + Sprachpakete: OCR fuer gescannte Dokumente (lokal, kein Cloud-OCR)
# poppler-utils: PDF -> Bild fuer den OCR-Pfad
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    curl \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness ohne Abhaengigkeiten -- ein kurzer Redis-Ausfall soll den Container
# nicht neu starten (die Bereitschaft prueft /api/health).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health/live || exit 1
