# Releases

Each release is a git tag on `main`. The recommender reports its release at `/recommend/health`, and
the web page shows it next to the title, linked to these notes. To release, add an entry at the top,
set `RELEASE` in [`src/recommender/__init__.py`](src/recommender/__init__.py) to the same tag, merge,
then tag the merge commit (`git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`).

## v0.0.2 (2026-10-22)

A more robust recommender service, with its prompts in files. Answers are unchanged: the prompts are
the same text as in v0.0.1.

### Changed

- **Retries on every LLM call.** Ollama and Bedrock calls are retried on throttling (429), a 5xx or a
  dropped connection, with exponential backoff and jitter; a `Retry-After` header sets the wait. Other
  errors fail at once. Each retry is an `llm_retry` event in the trace. Set in the new `[llm]` section
  of `settings.toml`.
- **Prompts in files.** Every recommender prompt is a Jinja2 template in
  [`src/recommender/prompts/`](src/recommender/prompts), with the tool descriptions (`tools.json`) and
  few-shot examples next to them. They ship in the wheel; `[paths] prompt_overrides` names a folder
  whose templates replace the packaged ones, published with the library.
- **Checked model replies.** The fixed JSON reply shapes are Pydantic models
  ([`llm_schemas.py`](src/recommender/llm_schemas.py)) that produce the schema sent to the model and
  validate what comes back.
- **Validated settings.** A bad value in `settings.toml` stops the start with the setting's name.
- **Web service.** Answers run on their own worker threads (`asyncer`), and a pipeline returns to the
  pool only when its answer is done, even if the user has left.
- **Release shown on the page.** `/recommend/health` reports the release, and the page shows it.

### Upgrade

- AWS: `just aws-recommender-deploy`. No infrastructure change, and no need to republish the library.
- Locally: restart `just recommender-serve`.

## v0.0.1 (2026-10-21)

The first deployed version: the AniList anime recommender on AWS.

- **Recommender** on EC2 behind CloudFront (`/recommend/*`), answering with `qwen.qwen3-32b` on
  Bedrock (bedrock-mantle). The built library is published to S3 as a versioned bundle and pulled at
  start.
- **Web page** on S3 and CloudFront, with passwordless Cognito sign-in (email codes, university
  domains only); an optional custom domain.
- **Chat Lambda** (`/api/chat`) with `google.gemma-3-4b-it`, still deployed alongside.
- **Local stack:** LocalStack, Ollama and Phoenix tracing; Arize AX tracing on AWS.
- **Evaluation** of factual and recommendation answers, the project report, and a monthly cost
  estimate ([costs](docs/infra/costs.md)).
