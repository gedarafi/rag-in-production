from __future__ import annotations

import argparse
import shutil
import socket
from pathlib import Path

import gradio as gr

from .config import Settings, load_settings
from .rag import RagPipeline


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
  #conversation {height: 50vh !important; min-height: 280px;}
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
    pipeline = RagPipeline(settings)

    def ask_question(question: str, history: list[dict] | None):
        history = list(history or [])
        if not question.strip():
            yield history, question, "Enter a question about your documents."
            return
        history.append({"role": "user", "content": question.strip()})
        history.append({"role": "assistant", "content": "Thinking..."})
        yield history, "", "Preparing documents and waiting for the model. The first answer can take longer."
        try:
            reply = pipeline.ask(question.strip())
            if not reply.strip():
                raise RuntimeError("The model returned an empty answer. Try asking again.")
            history[-1] = {"role": "assistant", "content": reply}
            yield history, "", "Ready."
        except (OSError, RuntimeError, ValueError) as error:
            history[-1] = {"role": "assistant", "content": f"Unable to answer: {error}"}
            yield history, "", "Unable to answer. See the message above."

    def rebuild_index() -> str:
        try:
            return f"Ready. Indexed {pipeline.ingest()} document chunks."
        except (OSError, RuntimeError, ValueError) as error:
            return f"Could not build the index: {error}"

    with gr.Blocks(title="Local RAG") as app:
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
                with gr.Accordion("Index & document settings", open=False):
                    gr.Markdown(
                        "The index is built on your first question. "
                        "After adding more files, rebuild it to include them."
                    )
                    ingest_button = gr.Button("Rebuild index", size="sm")
                status = gr.Markdown("Your library is ready for documents.", elem_id="document-status")
            with gr.Column(scale=1, min_width=320, elem_id="chat-panel"):
                with gr.Row(elem_id="chat-header"):
                    gr.Markdown("### Chat with your documents", scale=1)
                    clear_button = gr.Button("New chat", size="sm", scale=0, min_width=90)
                chat = gr.Chatbot(
                    label="Conversation",
                    show_label=False,
                    height="min(58vh, 620px)",
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

        add_button.click(
            lambda files: copy_uploaded_documents(files, settings),
            inputs=upload,
            outputs=status,
        )
        ingest_button.click(rebuild_index, outputs=status)
        for event in (ask_button.click, question.submit):
            event(
                ask_question,
                inputs=[question, chat],
                outputs=[chat, question, chat_status],
                concurrency_id="chat",
                concurrency_limit=1,
            )
        clear_button.click(
            lambda: ([], "", "Ready. Ask a question to begin."),
            outputs=[chat, question, chat_status],
            concurrency_id="chat",
            concurrency_limit=1,
        )

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