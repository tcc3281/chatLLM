FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy ONLY dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (they will be cached unless pyproject.toml/uv.lock changes)
RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-dev; \
    else \
        uv sync --no-install-project --no-dev; \
    fi

# Copy the rest of the source code
COPY . .

# Final sync to install the project itself (fast since deps are cached)
RUN uv sync --no-dev

EXPOSE 7860

# Use uv run to ensure the virtualenv is used correctly
CMD ["uv", "run", "main.py"]
