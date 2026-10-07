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

```powershell
uv sync
ollama pull llama3.2:3b
```

Put `.txt`, `.md`, `.pdf`, or `.docx` files in `data/documents`, then build the index and ask a
question:

```powershell
uv run rag ingest
uv run rag ask "What is in my documents?"
```

To launch the browser interface:

```powershell
uv run python -m src.gui
```

Open `http://127.0.0.1:7860`. If no index exists, asking a question builds it automatically.
After adding documents, click **Rebuild index** to include them. The first ingest downloads the configured embedding model from
Hugging Face and caches it locally.

### Configuration

Edit `settings.toml` to change document and index paths, prompt, models, chunking, retrieval, and
generation settings. The environment variables `RAG_LLM_MODEL`, `RAG_EMBEDDING_MODEL`, and
`RAG_OLLAMA_URL` temporarily override model settings.

Use another TOML file with `--config path/to/settings.toml`. The provider boundary is
`src/providers.py`, so another local generation server can be integrated without
changing document loading, indexing, prompting, or the CLI.
