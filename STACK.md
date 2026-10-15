# Tech stack

What the project uses today, and what is planned. Versions and settings come from
[`pyproject.toml`](pyproject.toml) and [`settings.toml`](settings.toml).

## Code

| Tool | Purpose | Notes |
| --- | --- | --- |
| Python | Language | `>= 3.14` (pinned in `.python-version`) |
| [uv](https://docs.astral.sh/uv/) | Package and environment management | `uv sync` to install, `uv add <pkg>` to add a dependency; commit `uv.lock` |

### Main libraries

| Library | Used for |
| --- | --- |
| `sentence-transformers` | Embeddings (default `all-MiniLM-L6-v2`) |
| `numpy` | Vector index, stored in `data/index/` |
| `pypdf`, `python-docx` | Reading `.pdf` and `.docx` documents |
| `requests` | Calling the Ollama API |
| `gradio` | Browser interface (`uv run python -m src.gui`) |

## Local

| Tool | Purpose | Status |
| --- | --- | --- |
| [Ollama](https://ollama.com/) | Runs the LLM locally (default `llama3.2:3b` at `http://localhost:11434`) | In use |
| [Docker](https://www.docker.com/) | Same environment on macOS, Windows and Linux | Planned |

## Cloud

| Tool | Purpose | Status |
| --- | --- | --- |
| [AWS](https://aws.amazon.com/) | Production deployment target | Planned |
| [LocalStack](https://www.localstack.cloud/) | Emulates AWS locally for development and tests, with no AWS account or cost | Planned |

## AI tooling

| Tool | Used for |
| --- | --- |
| Claude Code | Development and code review; the `aws-core` plugin is enabled in [`.claude/settings.json`](.claude/settings.json) |
| GitHub Copilot | Pull request review |
