# Multi-stage build: only the virtualenv reaches the runtime image.
# Works with the classic builder (no buildx required); CI additionally uses
# buildx layer caching.
#
#   docker build -t noesis:0.1.0 .
#   docker run --rm noesis:0.1.0

FROM ghcr.io/astral-sh/uv:0.13.1 AS uv

# --- builder -----------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install non-editable so the venv copied to the runtime stage is self-contained.
RUN uv venv --python /usr/local/bin/python /opt/venv \
 && uv pip install --python /opt/venv/bin/python .

# --- runtime -----------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# uid/gid 1000 matches the host user so bind-mounted ./data, ./configs and
# ./results stay writable from the container.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TORCH_HOME=/home/app/.cache/torch \
    UV_CACHE_DIR=/home/app/.cache/uv \
    NOESIS_CONFIG_DIR=/app/configs \
    NOESIS_DATA_DIR=/app/data \
    NOESIS_RESULTS_DIR=/app/results

RUN mkdir -p /app /home/app/.cache \
 && chown -R app:app /app /home/app

WORKDIR /app
USER app

# Default entrypoint is the smoke check; override with any project command.
CMD ["python", "-m", "memory_rl.smoke"]
