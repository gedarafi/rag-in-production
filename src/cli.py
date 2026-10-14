from __future__ import annotations

import argparse
import json
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
                            help="Print router plan, context choice, tool trace and timing to stderr")
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
            print(f"Prepared {count} documents. Run rag ingest --kb {args.command.removeprefix('prepare-')} to rebuild its index.")
            return
        from .rag import RagPipeline
        if args.command == "ingest":
            print("Loading embedding model...", file=sys.stderr, flush=True)
        pipeline = RagPipeline(load_settings(args.config, args.knowledge_base))
        if args.command == "tool":
            tools = pipeline.anime_tools()
            if tools is None:
                raise ValueError("Run rag prepare-anilist first to create the structured catalog.")
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
            started = time.perf_counter()
            try:
                print(pipeline.ask(args.question))
            finally:
                if args.debug:
                    debug = {"question": args.question,
                             "elapsed_seconds": round(time.perf_counter() - started, 3),
                             "routing": getattr(pipeline, "last_route", {}),
                             "retrieval": getattr(pipeline, "last_retrieval", {})}
                    print(json.dumps(debug, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
