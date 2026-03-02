# ============================================================
#  LegalMind API — Dockerfile
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first — prevents pip resolving 915MB GPU build
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu
RUN PYTHONPATH=/install/lib/python3.11/site-packages python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"

# ── Stage 2: Runtime ──────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# ── FIX: Set PYTHONPATH so 'core', 'api' packages resolve ─
# Without this, Python can't find 'core.db', 'core.config' etc.
ENV PYTHONPATH=/app

RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
