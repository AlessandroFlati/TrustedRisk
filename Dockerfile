FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps (libgomp for faiss-cpu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition first to leverage Docker layer caching
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY apps/ ./apps/

# Install the package + runtime dependencies
RUN pip install --no-cache-dir -e .[deploy]

# Copy artifacts + fixtures (produced by porting script before docker build)
COPY data/ ./data/
COPY fixtures/ ./fixtures/
COPY scripts/ ./scripts/

# Phase 6.3 — Cloud Run multi-mode entrypoint.
#
# `AGENT_MODULE` selects which ASGI app to serve:
#   AGENT_MODULE=mcp_server.server:build_http_app  (factory; default)
#   AGENT_MODULE=apps.specialist_discharge.server:app
#   AGENT_MODULE=apps.specialist_acute.server:app
#   AGENT_MODULE=apps.specialist_evidence.server:app
#   AGENT_MODULE=apps.specialist_population.server:app
#   AGENT_MODULE=apps.specialist_pedi_mh.server:app
#   AGENT_MODULE=apps.specialist_pa.server:app
#   AGENT_MODULE=apps.specialist_scribe.server:app
#   AGENT_MODULE=apps.specialist_patient.server:app
#   AGENT_MODULE=apps.composer.server:app
#   AGENT_MODULE=apps.cds_hooks.server:app
#   AGENT_MODULE=apps.federation_partner.server:app
#   AGENT_MODULE=apps.alert_agent.server:app
#   AGENT_MODULE=apps.scheduler_agent.server:app
#   AGENT_MODULE=apps.playground.server:app
#
# Cloud Run sets PORT automatically. We default to 8080 for local runs.

ENV PORT=8080
ENV TRUSTEDRISK_HOST=0.0.0.0
ENV PYTHONPATH=/app/src

ENV AGENT_MODULE=mcp_server.server:build_http_app

EXPOSE 8080

# `--factory` is required when AGENT_MODULE points at a factory function
# (mcp_server.server:build_http_app). The specialist apps export `app`
# directly so --factory is OFF for them. We detect by checking whether
# AGENT_MODULE ends with `:build_http_app`.
CMD ["sh", "-c", "if [ \"${AGENT_MODULE##*:}\" = \"build_http_app\" ]; then \
    exec python -m uvicorn \"$AGENT_MODULE\" --factory --host \"$TRUSTEDRISK_HOST\" --port \"$PORT\"; \
else \
    exec python -m uvicorn \"$AGENT_MODULE\" --host \"$TRUSTEDRISK_HOST\" --port \"$PORT\"; \
fi"]
