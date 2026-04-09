# ─── Dockerfile ───────────────────────────────────────────────────────────────
# Builds the FastAPI container image.
#
# Base image: python:3.11-slim
#   - 'slim' = stripped-down Debian with only Python. Much smaller than
#     the full python:3.11 image (~150MB vs ~900MB).
#
# Why we need system packages:
#   tesseract-ocr      → OCR engine for reading ingredient lists (Phase 1)
#   tesseract-ocr-eng  → English language data for Tesseract
#   libgl1             → OpenCV runtime dependency on newer Debian images
#   libglib2.0-0       → Another OpenCV dependency
#   libpq-dev + gcc    → Needed to compile asyncpg (PostgreSQL async driver)
#
# Layer caching strategy:
#   We copy requirements.txt FIRST and pip install BEFORE copying app code.
#   This way, re-running docker build after changing only .py files reuses
#   the cached pip install layer — much faster rebuilds.

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# --no-install-recommends: skip optional packages → smaller image
# Clean up apt cache at the end → smaller image
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app.
# This is also the path mounted in docker-compose volumes.
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# Copy everything else after pip install (so code changes don't bust pip cache)
COPY . .

# ── Port ──────────────────────────────────────────────────────────────────────
# Document that the app listens on 8000 (docker-compose maps 8000:8000)
EXPOSE 8000

# ── Default command ───────────────────────────────────────────────────────────
# This is overridden by docker-compose 'command:' for development (adds --reload).
# Used as-is when building for production deployment.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
