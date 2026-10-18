# Testing

## Try the chat locally and see its traces

Needs the tools and the LocalStack auth token from [INFRA.md](INFRA.md#localstack-local-aws).
One command starts LocalStack and Phoenix, deploys the stack to LocalStack, and registers the user
if needed:

```sh
just local-up me@student.uu.se
```

It prints the page URL and a snippet for the browser console. Open the page, paste the snippet in
the DevTools console (F12) to sign the page in, then use the chat box. LocalStack's Cognito has
no email-code sign-in, so local users get a fixed password and this snippet stands in for the code
screens; `just session me@student.uu.se` prints a fresh one. You can also ask from the terminal:

```sh
just chat me@student.uu.se "What is in week 3?"
```

Open Phoenix at http://localhost:6006, project **rag-in-production**: every message is an `answer`
trace with a `bedrock.converse` LLM span holding the system and user messages, the reply, the model
and the token counts.

LocalStack answers every request with the tiny `qwen2.5:0.5b`, whatever model is configured, so
this checks the path and the traces, not the quality of the answers. When you're done:

```sh
just destroy   # remove the local stack
just down      # stop LocalStack and Phoenix
```

## Automated tests

The Python tests (pytest, in [`tests/`](tests) and [`infra/scripts/`](infra/scripts)) cover the
chat and pre-sign-up Lambdas with fake AWS clients, so they need no AWS and no LocalStack. The
OpenTofu modules have `tofu test` suites in `infra/modules/*/tests/`.

```sh
just test-unit        # module tests with mocked providers, plus the Python tests
just test-py          # only the Python tests, with coverage; extra args go to pytest
just test-py -k chat  # for example, only the chat Lambda tests
just lint             # formatting checks and validation of every module
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
[INFRA.md](INFRA.md#ci).
