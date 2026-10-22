# Releases

Each release is a git tag on `main`. The recommender reports its release at `/recommend/health`, and
the web page shows it next to the title, linked to these notes. To release, add an entry at the top,
set `RELEASE` in [`src/recommender/__init__.py`](src/recommender/__init__.py) to the same tag, merge,
then tag the merge commit (`git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`).

## v0.0.3 (2026-10-22)

A security agent in front of the recommender: small talk, off-topic and unsafe messages are answered
at once instead of running the whole pipeline. Anime questions get the same answers as in v0.0.2.

### Added

- **Security agent** ([`security_agent.py`](src/recommender/security_agent.py)). Rules run first, with
  no LLM call: whole-message greetings, thanks, goodbyes and "what can you do?" get a fixed reply;
  prompt injection, requests for secrets and SQL/code injection get one fixed refusal. Messages the
  rules let through get one small LLM classification (`security_review`, templates
  `security_review_system.j2` and `security_review.j2`, schema `SecurityReview`): anime questions and
  follow-ups go on to the recommender, off-topic, small talk and unsafe or harmful requests get the
  fixed reply for their category. The LLM picks only the category, never the text; if the call fails
  the message passes. `[security] llm_review = false` keeps only the rules.
- **English only.** A message in another language gets a request to ask in English; romaji and native
  titles inside an English question still pass.
- **In the traces.** A `security_review` span with `recommender.security_category` on every question,
  and `recommender.response_path == "security_agent"` on `recommender.ask` when the agent answered.

### Changed

- **One extra LLM call** for every message the rules don't decide, including anime questions (at most
  200 output tokens). `llm_calls` in the debug view counts it.

### Upgrade

- AWS: `just aws-recommender-deploy`. No infrastructure change, and no need to republish the library
  (`llm_review` defaults to on when `[security]` is missing).
- Locally: restart `just recommender-serve`.

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
- **Release in the traces.** Every span's metadata has `release`, so Phoenix can filter on
  `metadata["release"] == "v0.0.2"`; the trace resource has it as `service.version`.

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
