from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import KNOWLEDGE_BASES, load_settings, select_knowledge_base
from .datasets import prepare_anilist, prepare_multidoc2dial
from .vndb import prepare_vndb
from .progress import TerminalProgress


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local, configurable RAG pipeline.")
    parser.add_argument("--config", default="settings.toml", help="Path to the TOML settings file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select-kb", help="Select the knowledge base used by the GUI and CLI defaults")
    select_parser.add_argument("name", choices=KNOWLEDGE_BASES)
    ingest_parser = subparsers.add_parser("ingest", help="Read documents and rebuild the local vector index")
    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--debug", action="store_true",
                            help="Print router plan, context choice, tool trace and timing to stderr "
                                 "(the stage-by-stage trace goes to Phoenix)")
    setup_parser = subparsers.add_parser("setup", help="Build the AniList library once (what the GUI setup button does)")
    setup_parser.add_argument("--source", type=Path, help="Local AniList CSV or ZIP instead of downloading it")
    setup_parser.add_argument("--rebuild", action="store_true", help="Rebuild prepared data and indexes")
    publish_parser = subparsers.add_parser("publish", help="Upload the built AniList library to S3 as the current version")
    pull_parser = subparsers.add_parser("pull", help="Download the current published library from S3")
    for command_parser in (publish_parser, pull_parser):
        command_parser.add_argument("--bucket", default=os.environ.get("RECOMMENDER_BUCKET"),
                                    help="S3 bucket (default: RECOMMENDER_BUCKET)")
        command_parser.add_argument("--prefix", default=os.environ.get("RECOMMENDER_PREFIX", "recommender"))
    pull_parser.add_argument("--target", type=Path, default=Path(os.environ.get("RECOMMENDER_DATA_DIR", "/var/lib/recommender")))
    pull_parser.add_argument("--version", help="A published version instead of the current one")
    serve_parser = subparsers.add_parser("serve", help="Serve signed-in web users (needs the server extra)")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    tool_parser = subparsers.add_parser("tool", help="Run a local AniList catalog tool without an LLM")
    tool_parser.add_argument("tool", choices=("popular", "recommend", "title", "character"))
    tool_parser.add_argument("query", nargs="?", default="")
    tool_parser.add_argument("--limit", type=int, default=5, choices=range(1, 21), metavar="1-20")
    tool_parser.add_argument("--kb", dest="knowledge_base", choices=("anilist",), default="anilist")
    for command_parser in (ingest_parser, ask_parser):
        command_parser.add_argument("--knowledge-base", "--kb", choices=KNOWLEDGE_BASES)
    for name in ("anilist", "multidoc2dial", "vndb"):
        prepare_parser = subparsers.add_parser(f"prepare-{name}", help=f"Prepare {name} documents")
        prepare_parser.add_argument("--source", type=Path, help="Local dataset file (dump archive or extracted directory for VNDB)")
        if name == "vndb":
            prepare_parser.add_argument("--spoiler-level", type=int, choices=(0, 1, 2), default=0,
                                        help="Maximum database spoiler flag: 0 none, 1 minor, 2 major")
            prepare_parser.add_argument("--reviews", type=Path, help="Local provenance-bearing review JSONL file")
    args = parser.parse_args()

    try:
        if args.command == "select-kb":
            select_knowledge_base(args.config, args.name)
            print(f"Selected {args.name}. Restart the GUI to use this knowledge base.")
            return
        if args.command.startswith("prepare-"):
            prepare = {"prepare-anilist": prepare_anilist, "prepare-multidoc2dial": prepare_multidoc2dial, "prepare-vndb": prepare_vndb}[args.command]
            options = {"spoiler_level": args.spoiler_level, "reviews": args.reviews} if args.command == "prepare-vndb" else {}
            count = prepare(load_settings(args.config).root, args.source, **options)
            print(f"Prepared {count} documents. Run recommender ingest --kb {args.command.removeprefix('prepare-')} to rebuild its index.")
            return
        if args.command == "setup":
            setup(args.config, args.source, args.rebuild)
            return
        if args.command in ("publish", "pull"):
            from . import bundle
            if not args.bucket:
                raise ValueError("Pass --bucket or set RECOMMENDER_BUCKET.")
            if args.command == "publish":
                print("Packing and uploading the library; this takes a few minutes.", file=sys.stderr, flush=True)
                manifest = bundle.publish(load_settings(args.config, "anilist"), Path(args.config), args.bucket, args.prefix)
                print(f"Published {manifest['version']} ({manifest['bundle_size'] / 1e6:,.0f} MB) to "
                      f"s3://{args.bucket}/{args.prefix}/ and made it current.")
            else:
                release = bundle.pull(args.bucket, args.target, args.prefix, args.version)
                print(f"Serving {release.name} from {args.target / bundle.CURRENT}.")
            return
        if args.command == "serve":
            from .server import serve
            serve(args.host, args.port)
            return
        from .rag import RagPipeline
        if args.command == "ingest":
            print("Loading embedding model...", file=sys.stderr, flush=True)
        settings = load_settings(args.config, args.knowledge_base)
        pipeline = RagPipeline(settings)
        if args.command == "tool":
            tools = pipeline.anime_tools()
            if tools is None:
                raise ValueError("Run recommender prepare-anilist first to create the structured catalog.")
            if args.tool != "popular" and not args.query.strip():
                raise ValueError(f"The {args.tool} tool requires a query.")
            if args.tool == "popular":
                result = tools.listing(tools.popular(args.limit, tools.filters(args.query)),
                                       "Most popular in the imported AniList snapshot.")
            elif args.tool == "recommend":
                records, reasons = tools.recommend(args.query, args.limit)
                result = tools.listing(records, "Recommendations from the imported AniList snapshot.", reasons)
            else:
                from .query_router import QueryPlan
                plan = QueryPlan("character" if args.tool == "character" else "overview", args.query)
                result = tools.route(args.query, plan=plan) or "No matching entry found in the imported catalog."
            print(result)
            return
        if args.command == "ingest":
            print(f"Indexed {pipeline.ingest(progress=TerminalProgress())} chunks.")
            if args.knowledge_base:
                select_knowledge_base(args.config, args.knowledge_base)
                print(f"Selected {args.knowledge_base} as the default knowledge base. Restart the GUI to apply.")
        else:
            ask(pipeline, settings, args.question, args.debug)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def setup(config: str, source: Path | None, rebuild: bool) -> None:
    """Run every AniList setup stage in the terminal; later launches reuse the result."""
    from .anilist_setup import AniListWorkspace

    workspace = AniListWorkspace(load_settings(config, "anilist"))
    snapshot, printed = None, set()
    for snapshot in workspace.setup(str(source) if source else None, rebuild, progress=TerminalProgress()):
        for key, step in snapshot["steps"].items():
            if step["status"] in ("completed", "error") and key not in printed:
                printed.add(key)
                print(f"{'✓' if step['status'] == 'completed' else '✕'} {step['label']}: {step['message']}", flush=True)
    if snapshot is None or not snapshot["ready"]:
        raise RuntimeError(snapshot["message"] if snapshot else "Setup did not run.")
    print("Library ready. Start chatting with `just recommender-gui` or `just recommender-ask`.")


def ask(pipeline, settings, question: str, debug: bool) -> None:
    """Print the answer on stdout; the trace ID (and with --debug, the routing) on stderr."""
    from . import RELEASE
    from ..tracing import flush, setup_tracing, trace_id, tracer

    exporting = setup_tracing(settings.tracing_endpoint, settings.tracing_project, RELEASE)
    started = time.perf_counter()
    with tracer.start_as_current_span("recommender.cli", openinference_span_kind="chain") as span:
        try:
            print(pipeline.ask(question))
        finally:
            if debug:
                details = {"question": question,
                           "elapsed_seconds": round(time.perf_counter() - started, 3),
                           "routing": getattr(pipeline, "last_route", {}),
                           "retrieval": getattr(pipeline, "last_retrieval", {})}
                print(json.dumps(details, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
            if exporting:
                print(f"Trace {trace_id(span)} in Phoenix project {settings.tracing_project}", file=sys.stderr)
            flush()


if __name__ == "__main__":
    main()
