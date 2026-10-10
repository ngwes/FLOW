"""Static checks on the Docker assets; real builds run with -m docker (Task 23)."""

from pathlib import Path

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


def test_dockerignore_excludes_runtime_artifacts() -> None:
    lines = {
        line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    assert {"runs/", "data/", ".git/", "tests/"} <= lines
