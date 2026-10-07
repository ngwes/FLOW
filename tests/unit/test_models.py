import sys
from pathlib import Path

import pytest
import torch

from flaw.config import ModelSection
from flaw.models import ModelNotFound, build_model


def test_mnist_cnn_forward() -> None:
    model = build_model(ModelSection(name="mnist_cnn"))
    out = model(torch.zeros(2, 1, 28, 28))
    assert out.shape == (2, 10)
    assert out.dtype == torch.float32


def test_cifar_cnn_forward() -> None:
    model = build_model(ModelSection(name="cifar_cnn"))
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_build_returns_fresh_instances() -> None:
    a = build_model(ModelSection(name="mnist_cnn"))
    b = build_model(ModelSection(name="mnist_cnn"))
    assert a is not b


def test_state_dict_is_float_only() -> None:
    sd = build_model(ModelSection(name="mnist_cnn")).state_dict()
    assert all(t.is_floating_point() for t in sd.values())


def test_unknown_name_raises() -> None:
    with pytest.raises(ModelNotFound, match="nope"):
        build_model(ModelSection(name="nope"))


def test_import_path(tmp_path: Path) -> None:
    (tmp_path / "my_net.py").write_text(
        "import torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n\n"
        "    def forward(self, x):  # type: ignore[no-untyped-def]\n"
        "        return self.fc(x)\n",
        encoding="utf-8",
    )
    model = build_model(ModelSection(class_path="my_net.Net", module_path=str(tmp_path)))
    assert model(torch.zeros(1, 4)).shape == (1, 2)
    sys.path.remove(str(tmp_path))


def test_import_path_missing_module_raises() -> None:
    with pytest.raises(ModelNotFound, match="missing_module"):
        build_model(ModelSection(class_path="missing_module.Net"))


def test_import_path_without_dot_raises() -> None:
    with pytest.raises(ModelNotFound, match="nodot"):
        build_model(ModelSection(class_path="nodot"))
