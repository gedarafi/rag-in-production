# RAG in production

A course chatbot and an anime recommender, run locally and deployed to AWS:

- **Anime recommender** ([`src/recommender/`](src/recommender)): answers questions about anime and
  recommends titles from an AniList snapshot, with local tools and an LLM planner.
- **Local RAG** ([`src/local/`](src/local)): a minimal RAG over your own documents.
- **AWS deployment** ([`infra/`](infra), [`web/`](web)): a sign-in page with university email codes,
  a chat Lambda and the recommender on EC2, all on Bedrock; LocalStack for local testing.

## Quick start

Needs the tools and the LocalStack auth token from [infra](docs/infra/README.md#localstack-local-aws).
Start LocalStack and Phoenix, deploy the stack to it and register yourself:

```sh
just local-up me@student.uu.se
```

It prints the page URL and how to sign in; [testing](docs/testing.md) has the rest.

## Docs

| Doc | Contents |
| --- | --- |
| [Recommender](docs/recommender/README.md) | GUI, setup, how questions are answered, configuration, CLI, evaluation |
| [Local RAG](docs/local/README.md) | The minimal RAG over your own documents |
| [Infrastructure](docs/infra/README.md) | LocalStack, OpenTofu/Terragrunt, deploying to AWS, Arize AX, CI |
| [Testing](docs/testing.md) | Trying everything locally and on AWS, automated tests |
| [Tech stack](docs/stack.md) | Tools and services in use and planned |
| [Roadmap](docs/roadmap.md) | Current cost and next steps |
| [GDPR](docs/gdpr.md) | What is still needed before real students use it (not compliant yet) |

## Team rules

- Create a pull request from a new branch for each contribution. Never push directly to main.
- At least one approval from another team member is required before merging.
- AI assistance and code review are welcome; experimentation and collaboration are encouraged.

## GenAI used so far

- GitHub Copilot: pull request review.
- Claude (Anthropic): `/code-review --fix`, Claude Code and its VS Code plugin, and the
  [AWS Agent Toolkit](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/plugins.html)
  (`aws-core` plugin), enabled for everyone in `.claude/settings.json`. Claude Code offers to
  install it when you first trust the repository; otherwise run
  `/plugin install aws-core@claude-plugins-official`, then `/reload-plugins`.
