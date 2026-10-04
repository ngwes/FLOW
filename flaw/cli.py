"""FL@W command line interface."""

import typer

from flaw import __version__

app = typer.Typer(help="FL@W - Federated Learning at Work")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"flaw {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True),
) -> None:
    """Train one model across the PCs you already have."""


def main() -> None:
    app()
