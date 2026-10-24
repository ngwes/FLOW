import importlib

import pytest
from typer.testing import CliRunner

import flaw
from flaw.cli import app


def test_version_attribute() -> None:
    assert flaw.__version__ == "0.2.0"


def test_cli_version_flag() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "flaw 0.2.0" in result.output


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_main_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    from flaw import cli

    called: list[int] = []
    monkeypatch.setattr(cli, "app", lambda: called.append(1))
    cli.main()
    assert called == [1]


# Coverage no longer excludes `if __name__ == "__main__":` blocks by default
# (coverage 7.16), so execute the guard for real instead of excluding it. Kept
# before `test_main_module_importable` so `flaw.__main__` is not yet in
# sys.modules when runpy executes it (runpy warns about pre-imported modules).
def test_main_module_runs_as_main(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    from flaw import cli

    called: list[int] = []
    monkeypatch.setattr(cli, "main", lambda: called.append(1))
    runpy.run_module("flaw.__main__", run_name="__main__")
    assert called == [1]


def test_main_module_importable() -> None:
    module = importlib.import_module("flaw.__main__")
    assert callable(module.main)
