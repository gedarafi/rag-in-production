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
* [AWS Agent Toolkit](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/plugins.html)
  (`aws-core` plugin): enabled for everyone in `.claude/settings.json`. Claude Code offers to
  install it when you first trust the repository. To install it manually, run
  `/plugin install aws-core@claude-plugins-official`, then `/reload-plugins`.

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

## Infrastructure

Everything lives in [`infra/`](infra):

| Path | Contents |
| --- | --- |
| `infra/docker-compose.yml` | LocalStack, which emulates AWS locally |
| `infra/modules/s3-bucket/` | OpenTofu module for a private, versioned, encrypted S3 bucket, with tests in `tests/` |
| `infra/live/` | Terragrunt config; `live/local/` deploys the modules to LocalStack |

Requires [Docker](https://www.docker.com/), [just](https://just.systems/), the
[AWS CLI](https://aws.amazon.com/cli/), [OpenTofu](https://opentofu.org/) and
[Terragrunt](https://terragrunt.gruntwork.io/). The versions are pinned in
`infra/.opentofu-version` and `infra/.terragrunt-version`; [tenv](https://github.com/tofuutils/tenv)
picks them up automatically.

### LocalStack (local AWS)

Create a free account at [app.localstack.cloud](https://app.localstack.cloud), copy your auth
token from **Auth Tokens**, and put it in `infra/.env`:

```sh
cp infra/.env.example infra/.env   # then set LOCALSTACK_AUTH_TOKEN
just up                            # start LocalStack and wait until it is healthy
just status                        # list the emulated services
just aws s3 ls                     # any AWS CLI command, sent to LocalStack
just down                          # stop it
```

### OpenTofu and Terragrunt

```sh
just lint             # formatting checks and validation
just test-unit        # module tests with a mocked provider (no LocalStack)
just test-localstack  # creates and destroys a test bucket on LocalStack
just plan             # plan infra/live/local
just apply            # create the bucket rag-documents-local on LocalStack
just e2e              # apply, check the bucket with the AWS CLI, then destroy
just destroy          # remove everything from infra/live/local
```

Run `just` to see all recipes.

### CI

[`.github/workflows/infra.yml`](.github/workflows/infra.yml) runs on pull requests that touch
`infra/`. The first job runs `just lint` and `just test-unit`. The second job starts LocalStack
and runs `just test-localstack` and `just e2e`; it needs a `LOCALSTACK_AUTH_TOKEN` repository
secret (**Settings → Secrets and variables → Actions**) and is skipped without it, for example
on pull requests from forks.
