from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import gradio as gr

from .config import KNOWLEDGE_BASES, Settings, load_settings, select_knowledge_base
from .rag import RagPipeline


def print_debug(snapshot: dict) -> None:
    """Mirror GUI request diagnostics to the launching terminal immediately."""
    print(json.dumps({'gui_debug': snapshot}, ensure_ascii=False, indent=2),
          file=sys.stderr, flush=True)


GUI_CSS = """
.gradio-container {max-width: 1240px !important; margin: auto; padding: 28px !important;}
#app-header {margin-bottom: 16px;}
#app-header h1 {font-size: 28px; letter-spacing: -0.8px; margin-bottom: 4px;}
#app-header p {color: var(--body-text-color-subdued);}
#workspace {gap: 24px; align-items: flex-start;}
#documents-panel {padding: 20px; border: 1px solid var(--border-color-primary); border-radius: 16px; background: var(--block-background-fill);}
#documents-panel h3 {margin: 0 0 4px;}
#chat-panel {gap: 12px; min-width: 0;}
#chat-header {align-items: center; gap: 12px;}
#chat-header h3 {margin: 0;}
#conversation {border-radius: 16px; overflow: hidden;}
#composer {align-items: flex-end; gap: 10px;}
#composer textarea {font-size: 15px;}
#send-button {min-height: 48px; border-radius: 12px;}
#chat-status {font-size: 12px; color: var(--body-text-color-subdued); min-height: 22px;}
#document-status {font-size: 13px;}
@media (max-width: 760px) {
  .gradio-container {padding: 14px !important;}
  #workspace {gap: 16px;}
  #documents-panel {flex-basis: 100% !important; width: 100% !important;}
  #conversation {height: 65vh !important; min-height: 320px;}
}
"""


def copy_uploaded_documents(files: list[str] | None, settings: Settings) -> str:
    if not files:
        return "No files selected."
    settings.documents.mkdir(parents=True, exist_ok=True)
    copied = 0
    for file_name in files:
        source = Path(file_name)
        if source.suffix.lower() in settings.extensions:
            shutil.copy2(source, settings.documents / source.name)
            copied += 1
    if copied == 0:
        return f"No supported files found. Supported types: {', '.join(settings.extensions)}"
    return f"Added {copied} document(s). Click Rebuild index before asking questions."


def build_app(config_path: str = "settings.toml") -> gr.Blocks:
    settings = load_settings(config_path)
    pipelines: dict[str, RagPipeline] = {}
    default_base = next((name for name in KNOWLEDGE_BASES if name != "custom" and
                         settings.documents == settings.root / "data/documents" / name), "custom")

    def get_pipeline(name: str) -> RagPipeline:
        if name not in pipelines:
            pipelines[name] = RagPipeline(load_settings(config_path, name))
        return pipelines[name]

    def ask_question(question: str, history: list[dict] | None, knowledge_base: str, latest_debug: dict):
        history = list(history or [])
        if not question.strip():
            yield history, question, "Enter a question about your documents.", latest_debug, gr.update(value=latest_debug)
            return
        previous_history = list(history)
        history.append({"role": "user", "content": question.strip()})
        history.append({"role": "assistant", "content": "Thinking..."})
        pending = {'status': 'running', 'question': question.strip(), 'knowledge_base': knowledge_base}
        print_debug(pending)
        yield history, "", "Preparing documents and waiting for the model. The first answer can take longer.", pending, gr.update(value=pending)
        pipeline = None
        started = perf_counter()
        try:
            pipeline = get_pipeline(knowledge_base)
            reply = pipeline.ask(question.strip(), history=previous_history)
            if not reply.strip():
                raise RuntimeError("The model returned an empty answer. Try asking again.")
            history[-1] = {"role": "assistant", "content": reply}
            debug = {**pending, 'status': 'completed', **deepcopy(getattr(pipeline, 'last_route', {}))}
            debug['elapsed_seconds'] = round(perf_counter() - started, 6)
            debug['trace_id'] = getattr(pipeline, 'last_trace_id', '')
            print_debug({**pending, 'status': 'completed', 'elapsed_seconds': debug['elapsed_seconds'],
                         'llm_calls': debug.get('llm_calls'), 'trace_id': debug['trace_id']})
            yield history, "", "Ready.", debug, gr.update(value=debug)
        except (OSError, RuntimeError, ValueError) as error:
            history[-1] = {"role": "assistant", "content": f"Unable to answer: {error}"}
            debug = {**pending, 'status': 'error', 'error': str(error), **deepcopy(getattr(pipeline, 'last_route', {}))}
            debug['elapsed_seconds'] = round(perf_counter() - started, 6)
            debug['trace_id'] = getattr(pipeline, 'last_trace_id', '')
            print_debug(debug)
            yield history, "", "Unable to answer. See the message above.", debug, gr.update(value=debug)

    def rebuild_index(knowledge_base: str, progress=gr.Progress()) -> str:
        try:
            progress(0, desc="Loading embedding model")
            pipeline = get_pipeline(knowledge_base)
            return f"Ready. Indexed {pipeline.ingest(progress=lambda stage, done, total: progress((done, max(total, 1)), desc=stage))} document chunks."
        except (OSError, RuntimeError, ValueError) as error:
            return f"Could not build the index: {error}"

    with gr.Blocks(title="Local RAG") as app:
        debug_state = gr.State({})
        gr.Markdown(
            "# Local RAG\nYour documents. A conversation away.",
            elem_id="app-header",
        )
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=0, min_width=260, elem_id="documents-panel"):
                gr.Markdown("### Your library\nAdd the documents you want to ask about.")
                upload = gr.File(
                    label="Drop documents here",
                    file_count="multiple",
                    file_types=list(settings.extensions),
                    type="filepath",
                    height=160,
                )
                gr.Markdown("PDF, Word, Markdown, and text files")
                add_button = gr.Button("Add to library", variant="primary")
                with gr.Accordion("Advanced", open=False):
                    knowledge_base = gr.Dropdown(
                        choices=list(KNOWLEDGE_BASES), value=default_base, label="Knowledge base",
                        info="Changing the knowledge base starts a new chat.",
                    )
                    gr.Markdown(
                        "The index is built on your first question. "
                        "After adding more files, rebuild it to include them."
                    )
                    ingest_button = gr.Button("Rebuild index", size="sm")
                    show_debug = gr.Checkbox(value=False, label="Show router debug",
                                             info="Inspect the latest query's plan, context choice, tools and LLM call count.")
                status = gr.Markdown("Your library is ready for documents.", elem_id="document-status")
            with gr.Column(scale=1, min_width=320, elem_id="chat-panel"):
                with gr.Row(elem_id="chat-header"):
                    gr.Markdown("### Chat with your documents", scale=1)
                    clear_button = gr.Button("New chat", size="sm", scale=0, min_width=90)
                chat = gr.Chatbot(
                    label="Conversation",
                    show_label=False,
                    height="max(600px, 75vh)",
                    layout="bubble",
                    buttons=["copy"],
                    placeholder=(
                        "### What would you like to know?\n"
                        "Ask a question about your library to get started.\n\n"
                        "Try **Summarize my documents** or **What are the key takeaways?**"
                    ),
                    elem_id="conversation",
                )
                with gr.Row(elem_id="composer"):
                    question = gr.Textbox(
                        show_label=False,
                        placeholder="Ask about your documents...",
                        lines=1,
                        max_lines=5,
                        scale=1,
                        min_width=180,
                        container=False,
                    )
                    ask_button = gr.Button(
                        "Send", variant="primary", scale=0, min_width=85, elem_id="send-button"
                    )
                chat_status = gr.Markdown(
                    "Enter to send. Shift + Enter for a new line.", elem_id="chat-status"
                )
                debug_output = gr.JSON(label="Router debug", value={}, visible=False)

        def add_documents(files, name):
            return copy_uploaded_documents(files, load_settings(config_path, name))

        def switch_knowledge_base(name):
            select_knowledge_base(config_path, name)
            return [], "", None, f"Using {name}. Uploads and index rebuilds apply to this library.", "Ready. Ask a question to begin.", {}, gr.update(value={})

        add_button.click(
            add_documents,
            inputs=[upload, knowledge_base],
            outputs=status,
            concurrency_id="chat", concurrency_limit=1,
        )
        knowledge_base.change(
            switch_knowledge_base, inputs=knowledge_base,
            outputs=[chat, question, upload, status, chat_status, debug_state, debug_output],
            concurrency_id="chat", concurrency_limit=1,
        )
        ingest_button.click(rebuild_index, inputs=[knowledge_base], outputs=status,
                            concurrency_id="chat", concurrency_limit=1)
        for event in (ask_button.click, question.submit):
            event(
                ask_question,
                inputs=[question, chat, knowledge_base, debug_state],
                outputs=[chat, question, chat_status, debug_state, debug_output],
                concurrency_id="chat",
                concurrency_limit=1,
            )
        clear_button.click(
            lambda: ([], "", "Ready. Ask a question to begin.", {}, gr.update(value={})),
            outputs=[chat, question, chat_status, debug_state, debug_output],
            concurrency_id="chat",
            concurrency_limit=1,
        )
        show_debug.change(lambda enabled, snapshot: gr.update(visible=enabled, value=snapshot),
                          inputs=[show_debug, debug_state], outputs=debug_output, queue=False)

    return app


def find_available_port(host: str, start_port: int) -> int:
    if not 1 <= start_port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    for port in range(start_port, 65536):
        try:
            with socket.create_server((host, port)):
                return port
        except OSError:
            continue
    raise OSError(f"No available port on {host} from {start_port} through 65535.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local RAG Gradio interface.")
    parser.add_argument("--config", default="settings.toml", help="Path to the TOML settings file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860, help="First port to try; defaults to 7860")
    parser.add_argument("--share", action="store_true", help="Ask Gradio for a public share URL")
    args = parser.parse_args()
    app = build_app(args.config)
    try:
        port = find_available_port(args.host, args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if port != args.port:
        print(f"Port {args.port} is unavailable. Using port {port}.")
    app.launch(
        server_name=args.host, server_port=port, share=args.share,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=GUI_CSS,
    )


if __name__ == "__main__":
    main()
