"""Static checks on the Docker assets; real builds run with -m docker (Task 23)."""

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_node_dockerfile_pins_base_and_entrypoint() -> None:
    text = (ROOT / "docker" / "node.Dockerfile").read_text(encoding="utf-8")
    assert "ARG FLAW_BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" in text
    assert 'ENTRYPOINT ["flaw"]' in text
    assert 'CMD ["node", "/etc/flaw/flaw.yaml"]' in text
    assert "FLAW_TORCH_INDEX_URL" in text
    assert "pytorch/pytorch:2.4.0-cpu" not in text


def test_orchestrator_dockerfile_is_cpu_only() -> None:
    text = (ROOT / "docker" / "orchestrator.Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "torchvision" in text
    assert "HEALTHCHECK" in text and "/status" in text


@pytest.mark.parametrize("dockerfile", ["node.Dockerfile", "orchestrator.Dockerfile"])
def test_dockerfile_copies_files_pyproject_declares(dockerfile: str) -> None:
    # `pip install .` reads `readme` and `license-files`; a missing file fails the build.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {project["readme"], *project["license-files"]}
    lines = (ROOT / "docker" / dockerfile).read_text(encoding="utf-8").splitlines()
    install = lines.index("RUN pip install .")
    copied = {
        name for line in lines[:install] if line.startswith("COPY ") for name in line.split()[1:-1]
    }
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert declared <= copied, f"{dockerfile} must COPY {sorted(declared - copied)}"
    assert not declared & ignored


def test_dockerignore_excludes_runtime_artifacts() -> None:
    lines = {
        line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    assert {"runs/", "data/", ".git/", "tests/"} <= lines
