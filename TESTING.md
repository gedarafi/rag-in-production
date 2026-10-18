# Testing

## Try the recommender locally and see its traces

Needs the tools and the LocalStack auth token from [INFRA.md](INFRA.md#localstack-local-aws).
Locally the recommender answers with a real model: LocalStack's Bedrock runs Llama 3.2 3B with
Ollama. With an NVIDIA GPU that takes seconds per answer, once Docker can use the GPU:

```sh
just gpu-setup   # once: installs the NVIDIA Container Toolkit, makes nvidia Docker's default runtime (asks for sudo)
```

LocalStack starts its model container itself and accepts no `--gpus` flag for it, hence the default
runtime; containers that don't ask for a GPU are unaffected. `just up` then asks for the GPU in that
container's environment by itself; without the toolkit the
model runs on CPU, which works but takes minutes per answer. One command starts LocalStack and
Phoenix, deploys the stack to LocalStack, publishes the full AniList catalog (the first time it
builds it from `data/`, downloading the dataset if needed) and registers the user if needed:

```sh
just local-up me@student.uu.se           # `just local-up me@student.uu.se fixture` for ten invented anime
```

It prints the page URL and a snippet for the browser console. Open the page, paste the snippet in
the DevTools console (F12) to sign the page in, then use the chat box. LocalStack's Cognito has
no email-code sign-in, so local users get a fixed password and this snippet stands in for the code
screens; `just session me@student.uu.se` prints a fresh one. You can also ask from the terminal:

```sh
just chat me@student.uu.se "What should I watch after Steins;Gate?"
just chat me@student.uu.se "Something shorter?" --conversation <conversation_id>
```

The question is queued (POST `/api/chat`), the worker Lambda answers it, and the page (or `just chat`)
polls `/api/chat/{job_id}` until the answer is there. Open Phoenix at http://localhost:6006, project
**rag-in-production**: each question is one trace, from the api Lambda's `api.ask` through the worker's
`recommender.job` down to every pipeline stage, model call, tool and retrieval (see
[Debugging with traces](README.md#debugging-with-traces)). `just recommender-logs` follows the
worker's log.

The first answer after a deploy also downloads the catalog bundle into the worker (about a minute for
the full catalog). Publishing the full catalog loads its 215k passages into LocalStack's Aurora, which
takes about ten minutes; the fixture loads in seconds.
Llama 3.2 3B is a small model, so answers are weaker than on AWS (Qwen3 32B), but the whole solution
runs: the same pipeline, prompts, tools and catalog.

CI answers with a deterministic stand-in instead (`RAG_LOCAL_MODEL_API=stub`, see below). Set it in
`infra/.env` (then `just apply`) for a quick local check of the plumbing; its answers are canned and
say nothing about quality. When you're done:

```sh
just destroy   # remove the local stack
just down      # stop LocalStack and Phoenix
```

## Automated tests

```sh
just test-unit         # module tests with mocked providers, plus both Python suites below
just test-py           # the Lambdas (api, sign-up) and shared modules, with coverage; extra args go to pytest
just test-recommender  # the recommender, with the worker image's dependencies (no PyTorch)
just lint              # formatting checks and validation of every module
```

The Python tests need no AWS, no LocalStack and no model: AWS clients are in-memory fakes
([`tests/fakes.py`](tests/fakes.py)) and the recommender's Bedrock calls go to a scripted transport
([`tests/recommender/bedrock_stub.py`](tests/recommender/bedrock_stub.py)). The OpenTofu modules have
`tofu test` suites in `infra/modules/*/tests/`.

These need a running stack:

```sh
just phoenix-up && just test-phoenix  # a question's trace (api Lambda, then worker) reaches Phoenix
just up && just test-localstack       # creates and destroys a test bucket on LocalStack
just apply && just test-vector-db     # the passage search in LocalStack's Aurora (pgvector, Data API)
                                      # finds what the file index finds (pytest -m localstack)
just e2e                              # deploys to LocalStack, publishes the fixture, checks the passage
                                      # search, sign-in, questions, a follow-up and isolation between
                                      # users, then destroys
RAG_LOCAL_MODEL_API=stub LOCALSTACK_BEDROCK_PREWARM=0 just restart e2e   # the same, as CI runs it
```

`just e2e` asks about the fixture's invented titles with whatever model the local stack uses. CI
sets `RAG_LOCAL_MODEL_API=stub`, so its answers are deterministic and no model is pulled.

`just test` runs lint, the unit tests and the LocalStack test. `just ci` runs the same steps as
CI from a fresh checkout's state: it runs `just clean` (deletes the OpenTofu and Terragrunt caches
and the Lambda build), then lint and the unit tests, then starts a fresh LocalStack for the
LocalStack test and `just e2e`, and stops it at the end. CI itself is described in
[INFRA.md](INFRA.md#ci).

Answer quality is measured with a real model by the evaluations (see the README) and on the
deployed AWS stack.
