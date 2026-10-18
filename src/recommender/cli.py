"""`recommender`: build, release and try the anime recommender.

  build      prepare the AniList catalog, index and dictionaries, then bundle them for release
  publish    upload a bundle to the artifacts bucket and make the worker serve it
  activate   serve an already published version (rollback)
  ask        answer one question in-process, with the Bedrock model named by MODEL_ID/MODEL_API
  tool       run a catalog tool without a model

ask sends its trace to TRACING_ENDPOINT (e.g. Phoenix from `just phoenix-up`) and prints the trace ID.
publish and activate use the usual AWS environment (AWS_PROFILE, or AWS_ENDPOINT_URL for LocalStack).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .config import KNOWLEDGE_BASES, load_settings, select_knowledge_base
from .datasets import prepare_anilist, prepare_multidoc2dial
from .vndb import prepare_vndb
from .progress import TerminalProgress

REPO = Path(__file__).resolve().parents[2]
BUNDLE_SETTINGS_KB = "anilist"


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recommender", description="Build, release and try the anime recommender.")
    parser.add_argument("--config", default="recommender.toml", help="Path to the TOML settings file")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Prepare the AniList library and bundle it for release")
    build.add_argument("--source", type=Path, help="Local AniList CSV or ZIP (default: data/raw, else download)")
    build.add_argument("--workspace", type=Path,
                       help="Folder to prepare the library in (default: the folder of --config)")
    build.add_argument("--rebuild", action="store_true", help="Rebuild documents, SQL and embeddings from the source")
    build.add_argument("--out", type=Path, default=Path("build/recommender"), help="Where to write <version>/")
    build.add_argument("--version", help="Bundle version (default: the current UTC time)")

    for name, text in (("publish", "Upload a bundle and serve it"), ("activate", "Serve a published version")):
        command = sub.add_parser(name, help=text)
        command.add_argument("target", help="Bundle folder (publish) or version (activate)")
        command.add_argument("--bucket", default=os.environ.get("ARTIFACT_BUCKET"), required=not os.environ.get("ARTIFACT_BUCKET"))
        command.add_argument("--parameter", default=os.environ.get("ARTIFACT_VERSION_PARAMETER"),
                             required=not os.environ.get("ARTIFACT_VERSION_PARAMETER"))
    sub.choices["publish"].add_argument("--no-activate", action="store_true", help="Upload only")

    ask = sub.add_parser("ask", help="Answer a question in-process with the Bedrock model from the environment")
    ask.add_argument("question")
    ask.add_argument("--knowledge-base", "--kb", choices=KNOWLEDGE_BASES)

    tool = sub.add_parser("tool", help="Run a local AniList catalog tool without an LLM")
    tool.add_argument("tool", choices=("popular", "recommend", "title", "character"))
    tool.add_argument("query", nargs="?", default="")
    tool.add_argument("--limit", type=int, default=5, choices=range(1, 21), metavar="1-20")
    tool.add_argument("--kb", dest="knowledge_base", choices=("anilist",), default="anilist")

    select = sub.add_parser("select-kb", help="Select the knowledge base used by ask and tool")
    select.add_argument("name", choices=KNOWLEDGE_BASES)
    ingest = sub.add_parser("ingest", help="Read documents and rebuild the vector index")
    ingest.add_argument("--knowledge-base", "--kb", choices=KNOWLEDGE_BASES)
    for name in ("anilist", "multidoc2dial", "vndb"):
        prepare = sub.add_parser(f"prepare-{name}", help=f"Prepare {name} documents")
        prepare.add_argument("--source", type=Path, help="Local dataset file (dump archive or extracted directory for VNDB)")
        if name == "vndb":
            prepare.add_argument("--spoiler-level", type=int, choices=(0, 1, 2), default=0,
                                 help="Maximum database spoiler flag: 0 none, 1 minor, 2 major")
            prepare.add_argument("--reviews", type=Path, help="Local provenance-bearing review JSONL file")
    return parser


def build(args) -> Path:
    from dataclasses import replace

    from .anilist_setup import AniListWorkspace
    from .artifacts import bundle

    config = Path(args.config).resolve()
    workspace = (args.workspace or config.parent).resolve()
    if workspace != config.parent:
        # A separate workspace (e.g. the LocalStack fixture) gets its own settings and prompts.
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, workspace / "recommender.toml")
        shutil.copytree(REPO / "prompts", workspace / "prompts", dirs_exist_ok=True)
    settings = load_settings(workspace / "recommender.toml", "anilist")
    setup = AniListWorkspace(replace(settings, root=workspace))
    progress = TerminalProgress()
    for snapshot in setup.setup(args.source, args.rebuild, progress):
        running = next((s for s in snapshot["steps"].values() if s["status"] in ("running", "error")), None)
        if running:
            print(f"{running['label']}: {running['status']} {running['message']}".rstrip(), file=sys.stderr, flush=True)
    if not setup.ready:
        raise RuntimeError(setup.message)
    print(setup.message, file=sys.stderr)
    text = (workspace / "recommender.toml").read_text(encoding="utf-8")
    import re
    text = re.sub(r'(?m)^(\s*default\s*=\s*)"[^"]*"', rf'\1"{BUNDLE_SETTINGS_KB}"', text, count=1)
    target = bundle(workspace, args.out.resolve(), args.version, settings_text=text)
    print(target)
    return target


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            build(args)
            return
        if args.command in ("publish", "activate"):
            from .artifacts import activate_version, publish
            if args.command == "publish":
                version = publish(Path(args.target), args.bucket, args.parameter, activate=not args.no_activate)
                print(f"Published {version}" + ("" if args.no_activate else " and made it active."))
            else:
                activate_version(args.target, args.bucket, args.parameter)
                print(f"{args.target} is now active.")
            return
        if args.command == "select-kb":
            select_knowledge_base(args.config, args.name)
            print(f"Selected {args.name}.")
            return
        if args.command.startswith("prepare-"):
            prepare = {"prepare-anilist": prepare_anilist, "prepare-multidoc2dial": prepare_multidoc2dial, "prepare-vndb": prepare_vndb}[args.command]
            options = {"spoiler_level": args.spoiler_level, "reviews": args.reviews} if args.command == "prepare-vndb" else {}
            count = prepare(load_settings(args.config).root, args.source, **options)
            print(f"Prepared {count} documents. Run recommender ingest --kb {args.command.removeprefix('prepare-')} to rebuild its index.")
            return
        from .rag import RagPipeline
        if args.command == "ingest":
            print("Loading embedding model...", file=sys.stderr, flush=True)
        pipeline = RagPipeline(load_settings(args.config, args.knowledge_base))
        if args.command == "tool":
            tools = pipeline.anime_tools()
            if tools is None:
                raise ValueError("Run recommender build first to create the structured catalog.")
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
                print(f"Selected {args.knowledge_base} as the default knowledge base.")
            return
        ask(pipeline, args.question)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def ask(pipeline, question: str) -> None:
    """Print the answer on stdout and the trace ID on stderr, where to look for the details."""
    from ..tracing import flush, setup_tracing, trace_id, tracer

    exporting = setup_tracing("recommender")
    with tracer.start_as_current_span("recommender.cli", openinference_span_kind="chain") as span:
        try:
            print(pipeline.ask(question))
        finally:
            if exporting:
                print(f"Trace {trace_id(span)} in project {os.environ.get('TRACING_PROJECT', 'recommender')}",
                      file=sys.stderr)
            flush()


if __name__ == "__main__":
    main()
