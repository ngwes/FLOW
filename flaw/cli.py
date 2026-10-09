"""FL@W command line interface."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from flaw import __version__
from flaw.config import ConfigError, FlawConfig, config_source_text, load_config
from flaw.node.runner import resolve_node_id, run_node
from flaw.orchestrator.server import create_app
from flaw.simulate import SimulationResult, prepare_run_dir, run_simulation

app = typer.Typer(help="FL@W - Federated Learning at Work", no_args_is_help=True)
console = Console()

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_CONFIG = 2


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"flaw {__version__}")
        raise typer.Exit()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_or_exit(path: Path) -> FlawConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(EXIT_BAD_CONFIG) from exc


@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Train one model across the PCs you already have."""
    _configure_logging(verbose)


@app.command()
def orchestrator(
    config: Path = typer.Argument(..., help="Training description YAML"),
    run_root: Path = typer.Option(Path("runs"), help="Where run artifacts are stored"),
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Start the orchestrator (server) for a training run."""
    cfg = _load_or_exit(config)
    run_dir = prepare_run_dir(cfg, run_root, snapshot_text=config_source_text(config))
    console.print(f"Run [bold]{cfg.run.name}[/bold] - state in {run_dir.run_dir}")
    console.print(f"Listening on http://{host}:{port}")
    uvicorn.run(create_app(cfg, run_dir), host=host, port=port, log_level="info")


@app.command()
def node(
    config: Path = typer.Argument(
        ..., help="Training description YAML (same file as the orchestrator's)"
    ),
    orchestrator_url: str = typer.Option(
        "http://127.0.0.1:8000", "--orchestrator", help="Orchestrator base URL"
    ),
) -> None:
    """Join a run as a training node (keeps running until the run ends or you stop it)."""
    cfg = _load_or_exit(config)
    console.print(f"Node [bold]{resolve_node_id(cfg)}[/bold] joining {orchestrator_url}")
    result = run_node(cfg, orchestrator_url.rstrip("/"))
    console.print(f"Node stopped: [bold]{result}[/bold]")
    raise typer.Exit(EXIT_FAILED if result == "fatal_error" else EXIT_OK)


@app.command()
def simulate(
    config: Path = typer.Argument(..., help="Training description YAML"),
    nodes: int = typer.Option(2, "--nodes", "-n", help="Number of simulated nodes"),
    rounds: int | None = typer.Option(None, "--rounds", help="Override federation.rounds"),
    timeout_s: float = typer.Option(600.0, "--timeout", help="Give up after this many seconds"),
    run_root: Path = typer.Option(Path("runs"), help="Where run artifacts are stored"),
) -> None:
    """Run an orchestrator plus N nodes on this machine (smoke test / demo)."""
    _load_or_exit(config)
    try:
        result = run_simulation(
            config, nodes, rounds=rounds, timeout_s=timeout_s, run_root=run_root
        )
    except (ValueError, TimeoutError, RuntimeError) as exc:
        console.print(f"[red]Simulation failed:[/red] {exc}")
        raise typer.Exit(EXIT_FAILED) from exc
    _print_result(result)


def _print_result(result: SimulationResult) -> None:
    table = Table(title=f"Run {result.run_dir.name}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("reason", str(result.reason))
    table.add_row("rounds", str(result.rounds))
    table.add_row("best round", str(result.best_round))
    best = f"{result.best_metric:.4f}" if result.best_metric is not None else "-"
    table.add_row("best metric", best)
    table.add_row("run dir", str(result.run_dir))
    console.print(table)


@app.command()
def status(
    orchestrator_url: str = typer.Option(
        "http://127.0.0.1:8000", "--orchestrator", help="Orchestrator base URL"
    ),
    timeout_s: float = typer.Option(5.0, "--timeout", help="HTTP timeout in seconds"),
) -> None:
    """Show a running orchestrator's progress and connected nodes."""
    url = orchestrator_url.rstrip("/") + "/status"
    try:
        response = httpx.get(url, timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        console.print(f"[red]Cannot reach {url}:[/red] {exc}")
        raise typer.Exit(EXIT_FAILED) from exc

    console.print(
        f"[bold]{body['name']}[/bold]  phase={body['phase']}  round={body['round']}  "
        f"best={body['best_metric']}@{body['best_round']}"
    )
    table = Table("node", "alive", "shard", "device", "last heartbeat (s)")
    for node_row in body["nodes"]:
        table.add_row(
            node_row["node_id"],
            str(node_row["alive"]),
            str(node_row["shard"]),
            node_row["device"],
            f"{node_row['last_heartbeat_age_s']:.1f}",
        )
    console.print(table)


def main() -> None:
    app()
