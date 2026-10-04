# FL@W dev/test container — the toolchain for `scripts/dev <command>`.
#
# Why this exists: the dev loop for FL@W needs Python, PyTorch and the test
# tooling. Baking them into an image keeps the host machine clean and gives
# every machine (and CI, if we ever want it) the same environment. Dependencies
# are installed here directly rather than from pyproject.toml so the image can
# be built before the project skeleton exists.
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8

# CPU-only torch: a tenth of the CUDA wheel size and all the tests need.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.4" \
    "torchvision>=0.19"

# Runtime + test tooling, floors mirrored from pyproject.toml.
RUN pip install \
    "fastapi>=0.111" \
    "uvicorn>=0.30" \
    "httpx>=0.27" \
    "pydantic>=2.7,<3" \
    "typer>=0.12" \
    "rich>=13.7" \
    "pyyaml>=6.0" \
    "pytest>=8.2" \
    "pytest-asyncio>=0.23" \
    "pytest-cov>=5" \
    "respx>=0.21" \
    "ruff>=0.5" \
    "mypy>=1.10" \
    "matplotlib>=3.8"

WORKDIR /app

# The repo is mounted at /app by `scripts/dev`; PYTHONPATH there makes
# `import flaw` work without an editable install (which Task 1 adds anyway).
