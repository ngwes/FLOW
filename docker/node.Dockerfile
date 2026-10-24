# GPU training node. Build for CUDA (default) or CPU-only (CI, no NVIDIA):
#   docker build -f docker/node.Dockerfile -t flaw-node:0.1.0 .
#   docker build -f docker/node.Dockerfile --build-arg FLAW_BASE=python:3.11-slim \
#     --build-arg FLAW_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t flaw-node:cpu .
# (pytorch/pytorch ships CUDA tags only; there is no :...-cpu base to point at.)
ARG FLAW_BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${FLAW_BASE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Optional CPU-only torch, used together with FLAW_BASE=python:3.11-slim.
# Empty (default) = keep the torch that came in the base image.
ARG FLAW_TORCH_INDEX_URL=
RUN if [ -n "$FLAW_TORCH_INDEX_URL" ]; then \
        pip install torch torchvision --index-url "$FLAW_TORCH_INDEX_URL"; \
    fi

COPY pyproject.toml README.md LICENSE ./
COPY flaw ./flaw
RUN pip install .

# The training description is expected at /etc/flaw/flaw.yaml (mounted, read-only).
ENTRYPOINT ["flaw"]
CMD ["node", "/etc/flaw/flaw.yaml"]
