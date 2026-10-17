from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Run a local, configurable RAG pipeline.", no_args_is_help=True)


@app.callback()
def settings(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option(help="Path to the TOML settings file")] = Path("settings.toml"),
) -> None:
    ctx.obj = config


def _run(ctx: typer.Context, action):
    """Build the pipeline from the settings file and run action on it, reporting errors briefly."""
    # Imported here so `rag --help` doesn't wait for the embedding libraries to load.
    from .config import load_settings
    from .rag import RagPipeline

    try:
        return action(RagPipeline(load_settings(str(ctx.obj))))
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command()
def ingest(ctx: typer.Context) -> None:
    """Read documents and rebuild the local vector index."""
    typer.echo(f"Indexed {_run(ctx, lambda pipeline: pipeline.ingest())} chunks.")


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="Question to answer from the indexed documents")],
) -> None:
    """Ask a question against the indexed documents."""
    typer.echo(_run(ctx, lambda pipeline: pipeline.ask(question)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
