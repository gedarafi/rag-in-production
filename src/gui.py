from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gradio as gr

from .config import Settings, load_settings
from .rag import RagPipeline


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

    def ask_question(question: str) -> str:
        if not question.strip():
            return "Enter a question about your documents."
        try:
            return pipeline.ask(question)
        except (OSError, RuntimeError, ValueError) as error:
            return f"Unable to answer: {error}"

    with gr.Blocks(title="Local RAG") as app:
        gr.Markdown("# Local RAG\nAsk questions about your local documents. The index is built automatically on your first question.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Documents")
                upload = gr.File(
                    label="Add documents",
                    file_count="multiple",
                    file_types=list(settings.extensions),
                    type="filepath",
                )
                add_button = gr.Button("Add files")
                ingest_button = gr.Button("Rebuild index", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Question",
                    placeholder="Ask something about your documents...",
                    lines=3,
                )
                ask_button = gr.Button("Ask", variant="primary")
                answer = gr.Markdown()

        add_button.click(
            lambda files: copy_uploaded_documents(files, settings),
            inputs=upload,
            outputs=status,
        )
        ingest_button.click(lambda: f"Indexed {pipeline.ingest()} chunks.", outputs=status)
        ask_button.click(ask_question, inputs=question, outputs=answer)
        question.submit(ask_question, inputs=question, outputs=answer)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local RAG Gradio interface.")
    parser.add_argument("--config", default="settings.toml", help="Path to the TOML settings file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Ask Gradio for a public share URL")
    args = parser.parse_args()
    build_app(args.config).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()