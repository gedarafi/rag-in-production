# Local RAG

A minimal Retrieval-Augmented Generation app over your own documents, in
[`src/local/`](../../src/local): Sentence Transformers for embeddings, Ollama for answers. It is
separate from the [anime recommender](../recommender/README.md).

## Setup

Install Python 3.14+, [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Ollama](https://ollama.com/download), and keep Ollama running (the app or `ollama serve`). From
the repository root:

```sh
uv sync
ollama pull llama3.2:3b
```

## Use it

Put `.txt`, `.md`, `.pdf` or `.docx` files in `data/documents/custom` (`[paths] documents` in
`settings.toml`), then:

```sh
uv run rag ingest                          # build the index in data/index/custom
uv run rag ask "What is in my documents?"
uv run python -m src.local.gui             # browser interface, http://127.0.0.1:7860
```

The first ingest downloads the embedding model from Hugging Face. Use `--config path/to/settings.toml`
for another settings file.

## Settings

`settings.toml` holds the paths, prompt, models, chunking, retrieval and generation settings.
`RAG_LLM_MODEL`, `RAG_EMBEDDING_MODEL` and `RAG_OLLAMA_URL` override the models.

- `embedding_device`: `auto` (CUDA, then MPS, then CPU), or force `cuda`, `mps` or `cpu`.
- `embedding_batch_size`: lower it if GPU memory runs out.
- `embedding_cache`: reuses unchanged chunks and lets an interrupted ingest resume. Bump
  `embedding_cache_version` to force fresh embeddings for the same model.

## Tracing

Start [Phoenix](https://phoenix.arize.com/) (needs `LOCALSTACK_AUTH_TOKEN` in `infra/.env`, see
[infra](../infra/README.md#localstack-local-aws)) and point the app at it:

```sh
just phoenix-up                                              # UI on http://localhost:6006
export RAG_PHOENIX_ENDPOINT=http://localhost:6006/v1/traces  # or [tracing] endpoint in settings.toml
uv run rag ask "What is in my documents?"                    # retrieved chunks, prompt and answer
```
