"""Static checks on the compose stack; the live smoke test lives in Task 23 (-m docker)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_compose_services_quotas_and_wiring() -> None:
    services = load("docker-compose.yml")["services"]
    assert set(services) == {"orchestrator", "node1", "node2"}

    assert "8000:8000" in services["orchestrator"]["ports"]
    assert services["orchestrator"]["restart"] == "unless-stopped"

    for name in ("orchestrator", "node1", "node2"):
        assert any(
            volume.endswith(":/etc/flaw/flaw.yaml:ro") for volume in services[name]["volumes"]
        ), f"{name} does not mount the training description at the conventional path"

    for name in ("node1", "node2"):
        node = services[name]
        limits = node["deploy"]["resources"]["limits"]
        assert limits["cpus"] == "4" and limits["memory"] == "8g"
        assert node["environment"]["FLAW_NODE_ID"] == name
        assert node["environment"]["FLAW_DATA_DIR"] == "/data"
        assert node["build"]["args"]["FLAW_TORCH_INDEX_URL"] == "${FLAW_NODE_TORCH_INDEX:-}"
        assert "http://orchestrator:8000" in node["command"]
        assert node["depends_on"]["orchestrator"]["condition"] == "service_healthy"


def test_gpu_override_pins_one_device_per_node() -> None:
    services = load("docker-compose.gpu.yml")["services"]
    for name, device_id in (("node1", "0"), ("node2", "1")):
        devices = services[name]["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["driver"] == "nvidia"
        assert devices[0]["device_ids"] == [device_id]
        assert devices[0]["capabilities"] == ["gpu"]
