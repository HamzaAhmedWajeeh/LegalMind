# ============================================================
#  LegalMind Frontend — Dockerfile (Streamlit)
#  Updated after full app.py implementation (Step 8)
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# curl: used by Docker health check
# wget: fallback for health probing in some environments
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit config — disable telemetry and set theme
RUN mkdir -p /app/.streamlit && cat > /app/.streamlit/config.toml << 'TOML'
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1a3c5e"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f4f8"
textColor = "#1a1a1a"
TOML

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
