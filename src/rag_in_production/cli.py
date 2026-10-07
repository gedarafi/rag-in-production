from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .rag import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local, configurable RAG pipeline.")
    parser.add_argument("--config", default="settings.toml", help="Path to the TOML settings file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Read documents and rebuild the local vector index")
    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents")
    ask_parser.add_argument("question")
    args = parser.parse_args()

    try:
        pipeline = RagPipeline(load_settings(args.config))
        if args.command == "ingest":
            print(f"Indexed {pipeline.ingest()} chunks.")
        else:
            print(pipeline.ask(args.question))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()