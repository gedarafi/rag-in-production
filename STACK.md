# Tech stack

What the project uses today, and what is planned. Versions and settings come from
[`pyproject.toml`](pyproject.toml) and [`settings.toml`](settings.toml).

## Code

| Tool | Purpose | Notes |
| --- | --- | --- |
| Python | Language | `>= 3.14` (pinned in `.python-version`) |
| [uv](https://docs.astral.sh/uv/) | Package and environment management | `uv sync` to install, `uv add <pkg>` to add a dependency; commit `uv.lock` |
| [just](https://just.systems/) | Command runner | Run `just` to list the recipes in [`justfile`](justfile) |

### Main libraries

| Library | Used for |
| --- | --- |
| `sentence-transformers` | Embeddings (default `all-MiniLM-L6-v2`) |
| `numpy` | Vector index, stored in `data/index/` |
| `pypdf`, `python-docx` | Reading `.pdf` and `.docx` documents |
| `requests` | Calling the Ollama API |
| `arize-phoenix-otel`, `openinference-instrumentation` | Sending traces of the RAG to Phoenix ([`src/tracing.py`](src/tracing.py)) |
| `gradio` | Browser interface (`uv run python -m src.gui`) |
| `typer` | Command-line interfaces: the `rag` command ([`src/cli.py`](src/cli.py)) and the LocalStack flow script ([`infra/scripts/local_flow.py`](infra/scripts/local_flow.py), run with `uv run --script`) |

## Local

| Tool | Purpose | Status |
| --- | --- | --- |
| [Ollama](https://ollama.com/) | Runs the LLM locally (default `llama3.2:3b` at `http://localhost:11434`) | In use |
| [Docker](https://www.docker.com/) | Same environment on macOS, Windows and Linux; runs LocalStack and Phoenix via [`infra/docker-compose.yml`](infra/docker-compose.yml) | In use |
| [Arize Phoenix](https://phoenix.arize.com/) | Traces of the local RAG (retrieved chunks, prompt, answer); `just phoenix-up`, UI on `http://localhost:6006` | In use (opt-in via `[tracing] endpoint`) |

## Cloud

| Tool | Purpose | Status |
| --- | --- | --- |
| [AWS](https://aws.amazon.com/) | Production deployment target ([`infra/live/aws/`](infra/live/aws)) | Sign-in and model-answered chat in use; RAG planned |
| Amazon Cognito | Passwordless sign-in with an email code, limited to university domains | In use |
| Amazon SES | Sends the sign-in codes | In use (sandbox until production access) |
| API Gateway (HTTP API) + Lambda | Chat endpoint behind a Cognito JWT authorizer; answers with a Bedrock model | In use |
| S3 + CloudFront | Hosts the page in [`web/`](web) and forwards `/api/*` to the API | In use |
| Amazon Bedrock | Model that answers the chat (Gemma 3 4B through Chat Completions on the OpenAI-compatible bedrock-mantle endpoint by default; LocalStack runs `qwen2.5:0.5b` through Converse instead); knowledge base for RAG | Model in use; knowledge base planned |
| [LocalStack](https://www.localstack.cloud/) | Emulates AWS locally for development and tests, with no AWS cost; needs a free auth token in `infra/.env` | In use |
| [OpenTofu](https://opentofu.org/) | Infrastructure as code (open-source Terraform fork); modules in [`infra/modules/`](infra/modules), tested with `tofu test` | In use (version in `infra/.opentofu-version`) |
| [Terragrunt](https://terragrunt.gruntwork.io/) | Wraps OpenTofu per environment; config in [`infra/live/`](infra/live) | In use (version in `infra/.terragrunt-version`) |

## CI/CD

| Tool | Purpose | Notes |
| --- | --- | --- |
| [GitHub Actions](https://docs.github.com/actions) | Lint and test the infrastructure on every pull request | [`.github/workflows/infra.yml`](.github/workflows/infra.yml); the LocalStack job needs the `LOCALSTACK_AUTH_TOKEN` repository secret |

## AI tooling

| Tool | Used for |
| --- | --- |
| Claude Code | Development and code review; the `aws-core` plugin is enabled in [`.claude/settings.json`](.claude/settings.json) |
| GitHub Copilot | Pull request review |
