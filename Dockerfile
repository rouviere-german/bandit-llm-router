# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="bandit-llm-router" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_DATA_DIR=/app/var \
    APP_MLFLOW_DIR=/app/var/mlflow \
    APP_PRICING_PATH=/app/config/pricing.toml
WORKDIR /app
# The arm catalog lives outside the package, so it ships as data: read-only, owned by root, and
# pointed at by an absolute APP_PRICING_PATH rather than the settings default (a path relative to
# the working directory). Without this the service cannot build its arms and exits on startup.
# The directory is created first and explicitly: letting COPY create it would apply --chmod to the
# directory too, and a 0444 directory cannot be traversed by the non-root runtime user.
RUN install --directory --owner=root --group=root --mode=0555 /app/config
COPY --chown=root:root --chmod=0444 config/pricing.toml /app/config/pricing.toml
# The service writes its decision log (DuckDB), tenant budget file, activation records and audit
# chain under APP_DATA_DIR. The root filesystem runs read-only, so that directory has to be a mount;
# creating it here, owned by the runtime user, makes Docker chown a fresh named volume to match.
RUN install --directory --owner=10001 --group=10001 --mode=0750 /app/var
VOLUME ["/app/var"]
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-m", "bandit_llm_router.core.healthcheck"]
ENTRYPOINT ["python", "-m", "bandit_llm_router"]
CMD ["serve"]
