# Orchestrator: CPU-only on purpose — it aggregates weights and (optionally)
# runs the central test-set evaluation; training happens on the nodes.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY flaw ./flaw
RUN pip install .

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/status', timeout=3)"

ENTRYPOINT ["flaw"]
CMD ["orchestrator", "/etc/flaw/flaw.yaml", "--host", "0.0.0.0"]
