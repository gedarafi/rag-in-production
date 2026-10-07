# Team rules

* Create a pull request from a new branch for each contribution.
  * Never push directly to main.
  * It requires at least one approval from another team member to merge.

* Feel free to use AI for anything, it is especially useful for code review.

* We embrace experimentation and collaboration; any idea is welcome.


# GenAI Used so far:

## GitHub Copilot

* Pull request review.

## Claude

* `/code-review --fix`

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
uv run python -m src.gui
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
uv run python -m src.gui
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
`src/providers.py`, so another local generation server can be integrated without
changing document loading, indexing, prompting, or the CLI.
