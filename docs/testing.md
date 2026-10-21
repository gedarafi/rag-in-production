# Testing

Needs the tools and the LocalStack auth token from [infra](infra/README.md#localstack-local-aws).
LocalStack's Cognito has no email-code sign-in, so local users get a fixed password and a snippet
or a generated `config.js` stands in for the code screens. Phoenix (http://localhost:6006) shows
the local traces; on AWS they go to Arize AX.

## Try the chat locally and see its traces

One command starts LocalStack and Phoenix, deploys the stack to LocalStack, and registers the user
if needed:

```sh
just local-up me@student.uu.se
```

It prints the page URL and a snippet for the browser console. Open the page, paste the snippet in
the DevTools console (F12) to sign the page in, then use the chat box;
`just session me@student.uu.se` prints a fresh snippet when the hour is up. You can also ask from
the terminal:

```sh
just chat me@student.uu.se "What is in week 3?"
```

In Phoenix, project **rag-in-production**, every message is an `answer` trace with a
`bedrock.converse` LLM span holding the system and user messages, the reply, the model and the
token counts.

LocalStack answers every request with the tiny `qwen2.5:0.5b`, whatever model is configured, so
this checks the path and the traces, not the quality of the answers. When you're done:

```sh
just destroy   # remove the local stack
just down      # stop LocalStack and Phoenix
```

## Try the anime recommender locally

The recommender's models run on [Ollama](https://ollama.com) (`ollama serve`, with the `llm` model
from `settings.toml` pulled); the recipes stop early with what to do when it isn't ready. Build the
library once (about 1 GB under `data/`; later runs reuse it while its inputs don't change):

```sh
just recommender-setup
```

Without the web stack, ask it directly or in its GUI (http://127.0.0.1:7861):

```sh
just recommender-ask "Something like Frieren"
just recommender-gui
```

As the deployed service, behind the sign-in, with the page calling it like CloudFront does on AWS:

```sh
just recommender-local-up              # LocalStack, apply, build and publish the library (once)
just signup me@student.uu.se           # once
just recommender-serve                 # terminal 1: the service on http://127.0.0.1:8080
just recommender-web me@student.uu.se  # terminal 2: the page on http://localhost:8000
```

`recommender-web` writes a `web/config.js` that signs the page in for an hour; run it again after
that. Open http://localhost:8000 and chat. Port 8080 is only the API: `/` there is a 404, and
`GET /recommend/health` says whether the library is loaded.

All three trace to Phoenix, project **recommender**: every question is a `recommender.ask` trace
with the agent's steps, the LLM calls and the retrieved passages. Restart `recommender-serve` after
a code change; it doesn't reload.

## Dev experience

The quickest way to try the recommender, with no LocalStack and no `just`: Python 3.14+,
[uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download) running.

```sh
uv sync
uv run anilist-gui   # the recommender on http://127.0.0.1:7861; click "Set up / resume"
```

## Check the deployment on real AWS

After `just aws-apply` (see [infra](infra/README.md#the-anime-recommender-on-ec2)):

1. `https://<url from just aws-outputs>/recommend/health` returns `"ready":true` once the instance
   has installed the package and pulled the library (a few minutes after a new instance).
2. Sign in on the page with an allowed email (real codes, by email) and ask something.
3. In [Arize AX](https://app.arize.com), project **rag-aws**, the question is a `recommender.ask`
   trace (chat Lambda messages are `answer` traces in the same project).

`aws-apply` changes only the infrastructure: the instance runs the wheel in its data bucket, so after
a code change run `just aws-recommender-deploy`, which builds, uploads and installs it. If something
is off, the service's log is on the instance; read it through SSM, e.g.

```sh
aws ssm start-session --target <instance_id>   # then: journalctl -u recommender -n 100
```

`Tracing off: could not read ...` there means the instance can't read the Arize key, and
`Failed to export spans ... 403` that Arize rejected the key or space ID.

## Automated tests

The Python tests (pytest, in [`tests/`](../tests) and [`infra/scripts/`](../infra/scripts)) cover the
chat and pre-sign-up Lambdas and the trace exporter with fake AWS clients, so they need no AWS and
no LocalStack. The recommender's tests in [`tests/recommender/`](../tests/recommender) need the
project's own dependencies (NumPy, PyTorch, Gradio) but no models: they use stubs. The OpenTofu
modules have `tofu test` suites in `infra/modules/*/tests/`.

```sh
just test-unit         # module tests with mocked providers, plus the Python tests
just test-py           # only the Python tests, with coverage; extra args go to pytest
just test-py -k chat   # for example, only the chat Lambda tests
just test-recommender  # the recommender's tests (not part of test-unit or CI)
just lint              # formatting checks and validation of every module
```

These need a running stack:

```sh
just phoenix-up && just test-phoenix  # the chat Lambda's traces reach Phoenix
just up && just test-localstack       # creates and destroys a test bucket on LocalStack
just e2e                              # deploys to LocalStack, checks sign-in and chat, then destroys
```

`just test` runs lint, the unit tests and the LocalStack test. `just ci` runs the same steps as
CI from a fresh checkout's state: it runs `just clean` (deletes the OpenTofu and Terragrunt caches
and the Lambda build), then lint and the unit tests, then starts a fresh LocalStack for the
LocalStack test and `just e2e`, and stops it at the end. CI itself is described in
[infra](infra/README.md#ci).
