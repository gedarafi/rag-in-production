"""AniList-only Gradio app with a complete local-library setup workflow."""
from __future__ import annotations

import argparse
from copy import deepcopy
from time import perf_counter

import gradio as gr

from .anilist_setup import AniListWorkspace
from .config import load_settings
from .gui import find_available_port, print_debug
from .progress import TerminalProgress

CSS = """
.gradio-container {max-width: 1440px !important; margin: auto; padding: 24px !important;}
#anilist-header {margin-bottom: 12px;}
#anilist-header h1 {font-size: 32px; letter-spacing: -1px;}
#setup-panel {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 18px;}
#anime-chat {border-radius: 16px;}
@media (max-width: 760px) {
  .gradio-container {padding: 12px !important;}
  #anime-chat {height: 65vh !important; min-height: 320px;}
}
"""


def render_steps(snapshot):
    icons = {'pending': '○', 'running': '◉', 'completed': '✓', 'ready': '✓', 'error': '✕'}
    rows = ['| Setup stage | Status | Time |', '| :-- | :-- | --: |']
    for step in snapshot['steps'].values():
        seconds = step['elapsed_seconds']
        elapsed = f'{seconds:.2f}s' if seconds is not None else '—'
        rows.append(f"| {step['label']} | {icons[step['status']]} {step['status']} | {elapsed} |")
    return '\n'.join(rows)


def build_app(config_path='settings.toml'):
    # This selection is explicit and does not change the multi-library app's default.
    settings = load_settings(config_path, 'anilist')
    workspace = AniListWorkspace(settings)

    def setup_workspace(source, rebuild, progress=gr.Progress()):
        terminal = TerminalProgress()
        def update(stage, done, total):
            terminal(stage, done, total)
            if total:
                progress((done, total), desc=stage)
            else:
                progress(None, desc=stage)
        for snapshot in workspace.setup(source, rebuild, progress=update):
            current = next(({'stage': key, **step} for key, step in snapshot['steps'].items()
                            if step['status'] in ('running', 'error')), None)
            print_debug({'app': 'anilist', 'setup': snapshot if current is None else current})
            yield (render_steps(snapshot), snapshot['message'],
                   gr.update(interactive=snapshot['ready']), gr.update(interactive=snapshot['ready']),
                   [], 'Ready. Ask about anime.' if snapshot['ready'] else 'Finish setup to enable chat.',
                   {}, gr.update(value={}))

    def ask_question(question, history):
        history = list(history or [])
        if not question.strip():
            yield history, question, 'Enter a question about anime.', {}, gr.update(value={})
            return
        previous = list(history)
        question = question.strip()
        history += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': 'Thinking...'}]
        pending = {'app': 'anilist', 'question': question, 'status': 'running'}
        print_debug(pending)
        yield history, '', 'Thinking…', pending, gr.update(value=pending)
        started = perf_counter()
        try:
            reply = workspace.ask(question, previous, debug=lambda event: print_debug({**pending, **event}))
            if not reply.strip():
                raise RuntimeError('The model returned an empty response. Please try again.')
            history[-1] = {'role': 'assistant', 'content': reply}
            debug = {**pending, **deepcopy(getattr(workspace.pipeline, 'last_route', {})), 'status': 'completed',
                     'elapsed_seconds': round(perf_counter() - started, 6)}
            print_debug({**pending, 'status': 'completed', 'elapsed_seconds': debug['elapsed_seconds'],
                         'llm_calls': debug.get('llm_calls')})
            yield history, '', 'Ready.', debug, gr.update(value=debug)
        except (OSError, RuntimeError, ValueError) as error:
            history[-1] = {'role': 'assistant', 'content': f'Unable to answer: {error}'}
            debug = {**pending, **deepcopy(getattr(workspace.pipeline, 'last_route', {})), 'status': 'error',
                     'error': str(error), 'elapsed_seconds': round(perf_counter() - started, 6)}
            print_debug(debug)
            yield history, '', 'Unable to answer. Check setup or the message above.', debug, gr.update(value=debug)

    with gr.Blocks(title='AniList Library') as app:
        debug_state = gr.State({})
        gr.Markdown('# AniList Library\nBuild your local anime library. Discover your next watch.', elem_id='anilist-header')
        with gr.Row():
            with gr.Column(scale=0, min_width=340, elem_id='setup-panel'):
                gr.Markdown('### Library setup\nSet up once, then chat. Existing downloads and embedding caches are reused.')
                steps = gr.Markdown(render_steps(workspace.snapshot()))
                setup_status = gr.Markdown(workspace.message)
                setup_button = gr.Button('Set up / resume', variant='primary')
                with gr.Accordion('Setup options', open=False):
                    source = gr.File(label='Optional local AniList CSV or ZIP', file_types=['.csv', '.zip'], type='filepath')
                    rebuild = gr.Checkbox(label='Rebuild prepared data and indexes', value=False,
                                          info='Reuses the raw dataset and cached embeddings. Starts a new chat.')
                    gr.Markdown(f'**Chat model:** `{settings.llm}`\n\n**Embeddings:** `{settings.embedding}`\n\n'
                                f'**Device:** `{settings.embedding_device}`\n\n'
                                'Model and device settings come from your TOML configuration. '
                                'Ollama must be installed and running. Setup downloads a missing chat model '
                                'and the embedding model automatically.')
                gr.Markdown('Dataset: the [AniList anime snapshot on Kaggle](https://www.kaggle.com/datasets/calebmwelsh/anilist-anime-dataset). '
                            'If downloading fails, supply the CSV or ZIP above and retry.')
            with gr.Column(scale=1, min_width=360):
                with gr.Row():
                    gr.Markdown('### Ask your anime library', scale=1)
                    clear = gr.Button('New chat', size='sm', scale=0)
                chat = gr.Chatbot(label='AniList conversation', show_label=False, height='max(600px, 72vh)',
                                  layout='bubble', buttons=['copy'], elem_id='anime-chat',
                                  placeholder='### Anime, characters, and recommendations\n'
                                  'Try **What is Assassination Classroom about?** or **Recommend five comedy anime.**')
                with gr.Row():
                    question = gr.Textbox(show_label=False, placeholder='Ask about anime…', lines=1, max_lines=5,
                                          scale=1, interactive=workspace.ready)
                    send = gr.Button('Send', variant='primary', scale=0, interactive=workspace.ready)
                chat_status = gr.Markdown('Ready. Ask about anime.' if workspace.ready else 'Finish setup to enable chat.')
                with gr.Accordion('Processing debug', open=False):
                    gr.Markdown('Stage progress also prints live in the terminal. The latest request snapshot appears below.')
                    debug_output = gr.JSON(label='Latest request', value={})

        setup_button.click(setup_workspace, inputs=[source, rebuild],
                           outputs=[steps, setup_status, question, send, chat, chat_status, debug_state, debug_output],
                           concurrency_id='anilist-workspace', concurrency_limit=1)
        for event in (send.click, question.submit):
            event(ask_question, inputs=[question, chat],
                  outputs=[chat, question, chat_status, debug_state, debug_output],
                  concurrency_id='anilist-workspace', concurrency_limit=1)
        clear.click(lambda: ([], '', 'Ready. Ask a question.', {}, gr.update(value={})),
                    outputs=[chat, question, chat_status, debug_state, debug_output],
                    concurrency_id='anilist-workspace', concurrency_limit=1)
    return app


def main():
    parser = argparse.ArgumentParser(description='Launch the AniList-only library setup and chat app.')
    parser.add_argument('--config', default='settings.toml')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    try:
        app = build_app(args.config)
        port = find_available_port(args.host, args.port)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if port != args.port:
        print(f'Port {args.port} is unavailable. Using {port}.', flush=True)
    app.launch(server_name=args.host, server_port=port, share=args.share,
               theme=gr.themes.Soft(primary_hue='violet', neutral_hue='slate'), css=CSS)


if __name__ == '__main__':
    main()
