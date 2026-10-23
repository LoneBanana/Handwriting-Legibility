# syntax=docker/dockerfile:1
# One image serving the API and the built web app on the same origin.
#   docker build -t legible .
#   docker run -p 8000:8000 -v legible-data:/data -v $PWD/runs:/models:ro \
#     -e LEGIBLE_JWT_SECRET=... -e LEGIBLE_HTR_CHECKPOINT=/models/htr-large/best.pt legible
# For GPU inference, swap the Python stage's base image for a CUDA runtime image and
# install the matching PyTorch wheel instead of the CPU one.

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[api,postgres]"
COPY --from=web /web/dist ./web/dist
RUN useradd --create-home --uid 10001 legible && mkdir -p /data && chown legible /data
USER legible
ENV LEGIBLE_ENVIRONMENT=production \
    LEGIBLE_DATA_DIR=/data \
    LEGIBLE_FRONTEND_DIST=/app/web/dist
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# One worker per process keeps a single copy of the model in memory; scale with replicas.
CMD ["uvicorn", "--factory", "legible.api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
