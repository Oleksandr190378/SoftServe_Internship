# Multi-stage Dockerfile for AI/ML Course Assistant
# Optimized for production with minimal image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ../requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Add user local bin to PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_LOGGER_LEVEL=info

# Copy application code
COPY .. .

# Create data directories with proper permissions
RUN mkdir -p data/chroma_db data/processed data/raw && \
    chmod -R 755 data/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import streamlit; print('healthy')" || exit 1

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "ui/app.py", "--logger.level=info"]
