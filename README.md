# Team rules

* Create a pull request from a new branch for each contribution.
  * Never push directly to main.
  * It requires at least one approval from another team member to merge.

* Feel free to use AI for anything, it is especially useful for code review.

* We embrace experimentation and collaboration; any idea is welcome.


# GenAI Used so far:

## GitHub Copilot

* Pull request review.

## Claude - Anthropic

* `/code-review --fix`
* [AWS Agent Toolkit](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/plugins.html)
  (`aws-core` plugin): enabled for everyone in `.claude/settings.json`. Claude Code offers to
  install it when you first trust the repository. To install it manually, run
  `/plugin install aws-core@claude-plugins-official`, then `/reload-plugins`.
* Claude Code
* Claude for VsCode Plugging

# Testing

## Try the chat locally

Needs the tools and the LocalStack auth token from [INFRA.md](INFRA.md#localstack-local-aws).
One command starts LocalStack and Phoenix, deploys the stack to LocalStack, and registers the user
if needed:

```sh
just local-up me@student.uu.se
```

How to sign in, where to see the traces, how to clean up and how to run the automated tests are
in [TESTING.md](TESTING.md).


---
# To Review

## Local RAG

This repository contains a configurable Retrieval-Augmented Generation application for local
documents. It uses Sentence Transformers for embeddings and Ollama for generation.

### Setup

Requires Python 3.14 or newer, `uv`, and [Ollama](https://ollama.com/).

#### Windows (PowerShell)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Ollama](https://ollama.com/download/windows), then start Ollama. From the repository root:

```powershell
uv sync
ollama pull llama3.2:3b
```

#### macOS (Terminal)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart Terminal so `uv` is available. Download [Ollama for macOS](https://ollama.com/download/mac),
install it in Applications, and launch it. Complete its first-run setup to enable the `ollama` command.
Keep Ollama running while using the app.

From the repository root, install Python and dependencies, download the model, and launch the GUI:

```bash
uv python install 3.14
uv sync
ollama pull llama3.2:3b
uv run python -m src.local.gui
```

If Ollama is installed but its server is not running, run `ollama serve` in a separate Terminal
window and leave it open. Skip this if the Ollama app is already running.

#### Using the app (Windows and macOS)

Put `.txt`, `.md`, `.pdf`, or `.docx` files in `data/documents`, then build the index and ask a
question:

```sh
uv run rag ingest
uv run rag ask "What is in my documents?"
```

To launch the browser interface:

```sh
uv run python -m src.local.gui
```

Open the URL printed in the terminal (normally `http://127.0.0.1:7860`). If the port is occupied,
the GUI automatically tries successive ports. Use `--port 8000` to start searching at another port. If no index exists, asking a question builds it automatically.
After adding documents, click **Rebuild index** to include them. The first ingest downloads the configured embedding model from
Hugging Face and caches it locally.

### Configuration

Edit `settings.toml` to change document and index paths, prompt, models, chunking, retrieval, and
generation settings. The environment variables `RAG_LLM_MODEL`, `RAG_EMBEDDING_MODEL`, and
`RAG_OLLAMA_URL` temporarily override model settings.

Use another TOML file with `--config path/to/settings.toml`. The provider boundary is
`src/local/providers.py`, so another local generation server can be integrated without
changing document loading, indexing, prompting, or the CLI.

### Tracing with Phoenix

[Arize Phoenix](https://phoenix.arize.com/) shows a trace of every `ingest` and question: the
retrieved chunks, the prompt sent to Ollama, and the answer.
It runs in Docker next to LocalStack, so the compose file still needs `LOCALSTACK_AUTH_TOKEN` in
`infra/.env` (see [LocalStack](INFRA.md#localstack-local-aws)):

```sh
just phoenix-up                                              # UI on http://localhost:6006
export RAG_PHOENIX_ENDPOINT=http://localhost:6006/v1/traces  # or set [tracing] endpoint in settings.toml
uv run rag ask "What is in my documents?"                    # the CLI and the GUI both send traces
just phoenix-down                                            # stop it; traces stay in infra/.phoenix/
```

Tracing is off while the endpoint is empty. The chat Lambda on LocalStack is traced too (see
[TESTING.md](TESTING.md#try-the-chat-locally-and-see-its-traces)); on AWS it sends its traces to Arize AX
(see [INFRA.md](INFRA.md#traces-in-arize-ax)).

## Infrastructure

The sign-in page, the chat API and their deployment to LocalStack and real AWS are described in
[INFRA.md](INFRA.md).


## Ethical considerations

* [Not GDPR compliant](GDPR.md)
