# Project tooling. Run `just` to list the recipes.

# Recipes run from infra/ so tenv picks up the pinned OpenTofu and Terragrunt versions.
set working-directory := "infra"
set dotenv-path := "infra/.env"

compose := "docker compose"
endpoint := env("LOCALSTACK_ENDPOINT", "http://localhost:4566")
region := env("AWS_DEFAULT_REGION", "us-east-1")
modules := "modules/s3-bucket modules/auth modules/chat-api modules/web modules/budget modules/recommender-host"
live := "live/local"
aws_live := env("RAG_AWS_LIVE", "live/aws") # `just aws-mantle` points the aws-* recipes at live/aws-mantle
aws_region := env("AWS_REGION", "eu-north-1")

# List the recipes
default:
    @just --list

# --- LocalStack ---

# Start LocalStack and wait until it is healthy
up:
    {{compose}} up -d --wait localstack

# Stop LocalStack
down:
    {{compose}} down

# Restart LocalStack
restart: down up

# Follow the LocalStack logs (`just logs --tail=all` to print and exit)
logs *args="-f":
    {{compose}} logs {{args}} localstack

# Show which emulated AWS services are available
status:
    @curl -fs {{endpoint}}/_localstack/health | python3 -m json.tool

# Run an AWS CLI command against LocalStack, e.g. `just aws s3 ls`
aws *args:
    AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test aws --endpoint-url {{endpoint}} --region {{region}} {{args}}

# Stop LocalStack and delete its saved state in infra/.localstack/
[confirm("Delete all LocalStack state in infra/.localstack/? (y/N)")]
reset: down
    rm -rf .localstack

# --- Phoenix (traces of the local RAG) ---

# Start Phoenix and wait until it is healthy
phoenix-up:
    {{compose}} up -d --wait phoenix
    @echo "Phoenix UI: http://localhost:6006 (send traces with RAG_PHOENIX_ENDPOINT=http://localhost:6006/v1/traces)"

# Stop Phoenix
phoenix-down:
    {{compose}} stop phoenix

# Follow the Phoenix logs (`just phoenix-logs --tail=all` to print and exit)
phoenix-logs *args="-f":
    {{compose}} logs {{args}} phoenix

# --- Anime recommender (local Ollama, traced in Phoenix) ---

phoenix_url := env("PHOENIX_URL", "http://localhost:6006")
recommender_project := env("RAG_PHOENIX_PROJECT", "recommender")
recommender_env := "RAG_PHOENIX_ENDPOINT=" + phoenix_url + "/v1/traces RAG_PHOENIX_PROJECT=" + recommender_project

# Ask the recommender one question and trace it in Phoenix, e.g. `just recommender-ask "Who is Subaru?" --debug`
[working-directory: '..']
recommender-ask question *args: phoenix-up _ollama-check
    {{recommender_env}} uv run recommender ask {{quote(question)}} --kb anilist {{args}}
    @echo "See the trace at {{phoenix_url}}, project {{recommender_project}}"

# Build the AniList library once (dataset, SQL catalog, embeddings); later GUI/ask runs reuse it. Pass --rebuild to redo it
[working-directory: '..']
recommender-setup *args: _ollama-check
    uv run recommender setup {{args}}

# Chat with the recommender in its GUI (http://127.0.0.1:7861), every question traced in Phoenix
[working-directory: '..']
recommender-gui *args: phoenix-up _ollama-check
    @echo "Traces: {{phoenix_url}}, project {{recommender_project}}"
    {{recommender_env}} uv run anilist-gui {{args}}

# --- Recommender service: library in S3, web users signed in with Cognito (src/recommender/server.py) ---

# Build the Python package the EC2 instance installs (dist/*.whl)
[working-directory: '..']
recommender-wheel:
    rm -f dist/rag_in_production-*.whl
    uv build --wheel --quiet
    @ls dist/*.whl

# Upload the built library to the LocalStack data bucket (after `just apply` and `just recommender-setup`)
[working-directory: '..']
recommender-publish:
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="$(terragrunt run --non-interactive --log-level error --working-dir infra/{{live}}/recommender-host -- output -raw bucket_name)"
    AWS_ENDPOINT_URL={{endpoint}} AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION={{region}} \
      uv run --quiet --extra server recommender publish --bucket "$bucket"

# One-time local setup: start LocalStack, deploy, build the library (reused if built) and publish it to S3
recommender-local-up: up apply recommender-setup recommender-publish
    @echo "Done. Sign up once with \`just signup you@student.uu.se\`, then run \`just recommender-serve\` and \`just recommender-web you@student.uu.se\`."

# Serve the published library on http://localhost:8080 for the local page, with Ollama for the models, traced in Phoenix
[working-directory: '..']
recommender-serve: phoenix-up _ollama-check
    #!/usr/bin/env bash
    set -euo pipefail
    out() { terragrunt run --non-interactive --log-level error --working-dir infra/{{live}}/$1 -- output -raw $2; }
    pool="$(out auth user_pool_id)"
    export AWS_ENDPOINT_URL={{endpoint}} AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION={{region}}
    export RECOMMENDER_BUCKET="$(out recommender-host bucket_name)" RECOMMENDER_DATA_DIR=data/serve
    # LocalStack signs tokens as its own issuer and serves the keys there.
    export COGNITO_ISSUER="${RECOMMENDER_LOCAL_ISSUER:-http://localhost.localstack.cloud:4566/$pool}"
    export COGNITO_JWKS_URL="{{endpoint}}/$pool/.well-known/jwks.json"
    export COGNITO_CLIENT_ID="$(out auth user_pool_client_id)" RECOMMENDER_CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
    export RECOMMENDER_WORKERS="${RECOMMENDER_WORKERS:-2}" {{recommender_env}}
    echo "Traces: {{phoenix_url}}, project {{recommender_project}}"
    uv run --quiet --extra server recommender serve --host 127.0.0.1 --port 8080

# Serve the page on http://localhost:8000 against `just recommender-serve`, e.g. `just recommender-web me@student.uu.se`
recommender-web email:
    #!/usr/bin/env bash
    set -euo pipefail
    client_id="$(terragrunt run --non-interactive --log-level error --working-dir {{live}}/auth -- output -raw user_pool_client_id)"
    domains="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1].split(",")))' "${RAG_ALLOWED_EMAIL_DOMAINS:-uu.se,student.uu.se}")"
    if ss -ltn | grep -q '127.0.0.1:8000 '; then echo "Port 8000 is in use; stop the other server first." >&2; exit 1; fi
    printf 'window.APP_CONFIG = {"region":"%s","userPoolClientId":"%s","allowedEmailDomains":%s,"apiBase":"","recommenderBase":"http://127.0.0.1:8080"};\n' \
      "{{region}}" "$client_id" "$domains" > ../web/config.js
    # LocalStack's Cognito has no email-code sign-in, so config.js signs the page in (for an hour).
    just session {{quote(email)}} | sed 's/ location.reload();$//' >> ../web/config.js
    echo "Open http://localhost:8000 (signed in as {{email}}; rerun this after an hour)"
    python3 -m http.server 8000 --bind 127.0.0.1 --directory ../web

# Upload the library and the package wheel to the AWS data bucket; the instance serves the new library on restart
[confirm("Upload the recommender library (about 1 GB) to your real AWS account? (y/N)")]
[working-directory: '..']
aws-recommender-publish: _aws-check recommender-wheel
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="$(terragrunt run --non-interactive --log-level error --working-dir infra/{{aws_live}}/recommender-host -- output -raw bucket_name)"
    aws s3 cp dist/rag_in_production-0.1.0-py3-none-any.whl "s3://$bucket/releases/" --region {{aws_region}}
    AWS_REGION={{aws_region}} uv run --quiet --extra server recommender publish --bucket "$bucket"

# Install the latest wheel on the instance and restart it on the current library (through SSM, no SSH)
[working-directory: '..']
aws-recommender-deploy: _aws-check recommender-wheel
    #!/usr/bin/env bash
    set -euo pipefail
    out() { terragrunt run --non-interactive --log-level error --working-dir infra/{{aws_live}}/recommender-host -- output -raw $1; }
    aws s3 cp dist/rag_in_production-0.1.0-py3-none-any.whl "s3://$(out bucket_name)/releases/" --region {{aws_region}}
    aws ssm send-command --region {{aws_region}} --instance-ids "$(out instance_id)" --document-name AWS-RunShellScript \
      --parameters commands=/usr/local/sbin/recommender-install --query Command.CommandId --output text

# Start or stop the recommender instance (stopped, it costs only its disk and Elastic IP)
aws-recommender-start: (_aws-recommender-power "start")
aws-recommender-stop: (_aws-recommender-power "stop")

_aws-recommender-power action: _aws-check
    #!/usr/bin/env bash
    set -euo pipefail
    id="$(terragrunt run --non-interactive --log-level error --working-dir {{aws_live}}/recommender-host -- output -raw instance_id)"
    aws ec2 {{action}}-instances --region {{aws_region}} --instance-ids "$id" --output text >/dev/null
    echo "Requested {{action}} of $id."

# Fail early when Ollama is down or the model in settings.toml isn't pulled
[working-directory: '..']
_ollama-check:
    #!/usr/bin/env bash
    set -euo pipefail
    url=$(sed -n 's/^ollama_url = "\(.*\)"/\1/p' settings.toml)
    model=$(sed -n 's/^llm = "\(.*\)"/\1/p' settings.toml)
    tags=$(curl -sf "$url/api/tags") || { echo "Ollama is not reachable at $url: start it with 'ollama serve'." >&2; exit 1; }
    grep -q "\"$model\"" <<<"$tags" || { echo "Ollama has no $model: run 'ollama pull $model'." >&2; exit 1; }

# --- OpenTofu / Terragrunt ---

# Format all OpenTofu and Terragrunt files
fmt:
    tofu fmt -recursive
    terragrunt hcl fmt --working-dir live

# Check formatting and validate every module and the Terragrunt config
lint:
    tofu fmt -check -recursive
    terragrunt hcl fmt --check --working-dir live
    for m in {{modules}}; do tofu -chdir=$m init -backend=false -input=false >/dev/null && tofu -chdir=$m validate || exit 1; done
    terragrunt hcl validate --working-dir live

# Delete the OpenTofu and Terragrunt caches and the Lambda build, as on a fresh checkout; the next init downloads the providers again
clean:
    find . \( -name .localstack -o -name .ollama \) -prune -o \( -name .terragrunt-cache -o -name .terraform \) -type d -prune -exec rm -rf {} +
    rm -rf .build

# Refresh the provider lock files for Linux, macOS and Windows
lock:
    for m in {{modules}}; do tofu -chdir=$m providers lock -platform=linux_amd64 -platform=linux_arm64 -platform=darwin_amd64 -platform=darwin_arm64 -platform=windows_amd64 || exit 1; done

# Unit tests with mocked providers, plus the Lambda unit tests (no LocalStack needed)
test-unit:
    for m in {{modules}}; do tofu -chdir=$m init -backend=false -input=false >/dev/null && tofu -chdir=$m test -filter=tests/unit.tftest.hcl || exit 1; done
    just test-py

# Python tests with coverage (the Lambda's chat dependency group, without the local RAG's models); extra args go to pytest
[working-directory: '..']
test-py *args:
    uv run --quiet --isolated --only-group test --only-group chat pytest --cov --cov-report=term-missing --ignore=tests/recommender {{args}}

# Anime recommender tests (they need the project's own dependencies: NumPy, Gradio, PyTorch); extra args go to pytest
[working-directory: '..']
test-recommender *args:
    uv run --quiet --extra server --group test pytest tests/recommender {{args}}

# Check that the chat Lambda's traces reach the running Phoenix (run `just phoenix-up` first)
[working-directory: '..']
test-phoenix:
    uv run --quiet --isolated --only-group test --only-group chat pytest -m phoenix -v

# Integration test that creates and destroys a bucket on LocalStack (run `just up` first)
test-localstack:
    tofu -chdir=modules/s3-bucket init -backend=false -input=false
    tofu -chdir=modules/s3-bucket test -filter=tests/localstack.tftest.hcl

# Run lint and every test
test: lint test-unit test-localstack

# Run the same steps as CI from a fresh checkout's state: lint and unit tests, then LocalStack tests and e2e on a fresh LocalStack
ci: clean lint test-unit
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'rc=$?; [ $rc -eq 0 ] || just logs --tail=all; just down; exit $rc' EXIT
    just up
    just status
    just test-localstack
    just e2e

# Plan the local environment
plan:
    terragrunt run --all --non-interactive --working-dir {{live}} -- plan

# Create the local environment on LocalStack
apply:
    terragrunt run --all --non-interactive --working-dir {{live}} -- apply -auto-approve

# Destroy the local environment
destroy:
    terragrunt run --all --non-interactive --working-dir {{live}} -- destroy -auto-approve

# Apply the local environment, check the bucket and the sign-in and chat flow, then destroy it
e2e:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'just destroy' EXIT
    just apply
    just aws s3api head-bucket --bucket rag-documents-local
    status="$(just aws s3api get-bucket-versioning --bucket rag-documents-local --query Status --output text)"
    [ "$status" = "Enabled" ] || { echo "expected versioning Enabled, got '$status'" >&2; exit 1; }
    echo "e2e: bucket rag-documents-local exists with versioning enabled"
    just flow e2e

# --- Real AWS (needs the "Real AWS" part of infra/.env and `aws sso login`) ---

# Refuse to run unless infra/.env is filled in and the AWS login is the expected account
_aws-check:
    #!/usr/bin/env bash
    set -euo pipefail
    for v in AWS_PROFILE RAG_AWS_ACCOUNT_ID RAG_SES_IDENTITY; do
      [ -n "${!v:-}" ] || { echo "Set $v in infra/.env (see infra/.env.example)." >&2; exit 1; }
    done
    account="$(aws sts get-caller-identity --query Account --output text)" \
      || { echo "Not signed in to AWS: run 'aws sso login --profile $AWS_PROFILE'." >&2; exit 1; }
    [ "$account" = "$RAG_AWS_ACCOUNT_ID" ] \
      || { echo "AWS_PROFILE $AWS_PROFILE is account $account, not RAG_AWS_ACCOUNT_ID $RAG_AWS_ACCOUNT_ID." >&2; exit 1; }

# Store (or rotate) the Arize API key the chat Lambda traces with, as an SSM SecureString; the Lambda reads it at cold start
aws-arize-key: _aws-check
    #!/usr/bin/env bash
    set -euo pipefail
    read -rsp "Arize API key (Space Settings > API Keys): " key; echo
    [ -n "$key" ] || { echo "No key given." >&2; exit 1; }
    # On stdin, so the key never appears in the process list or the shell history.
    printf '%s' "$key" | aws ssm put-parameter --region {{aws_region}} --name /rag-aws/arize-api-key \
      --type SecureString --overwrite --value file:///dev/stdin --query Version --output text \
      | sed 's/^/Stored \/rag-aws\/arize-api-key, version /'
    echo "Running Lambdas pick up a new key on their next cold start."

# Create the S3 bucket for the OpenTofu state (safe to run again)
aws-bootstrap: _aws-check
    #!/usr/bin/env bash
    set -euo pipefail
    bucket="rag-in-production-tfstate-$RAG_AWS_ACCOUNT_ID"
    if aws s3api head-bucket --bucket "$bucket" --region {{aws_region}} 2>/dev/null; then
      echo "State bucket $bucket already exists."
    elif [ "{{aws_region}}" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$bucket" --region us-east-1
    else
      aws s3api create-bucket --bucket "$bucket" --region {{aws_region}} \
        --create-bucket-configuration LocationConstraint={{aws_region}}
    fi
    aws s3api put-public-access-block --bucket "$bucket" --region {{aws_region}} \
      --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    aws s3api put-bucket-versioning --bucket "$bucket" --region {{aws_region}} --versioning-configuration Status=Enabled
    aws s3api put-bucket-encryption --bucket "$bucket" --region {{aws_region}} \
      --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
    echo "State bucket $bucket is ready."

# Plan the real AWS environment
aws-plan: _aws-check
    terragrunt run --all --non-interactive --working-dir {{aws_live}} -- plan

# Deploy to real AWS (CloudFront takes a few minutes)
[confirm("Deploy to your real AWS account? (y/N)")]
aws-apply: _aws-check
    terragrunt run --all --non-interactive --working-dir {{aws_live}} -- apply -auto-approve

# Page URL, API URL, user pool and client IDs of the real AWS environment
aws-outputs: _aws-check
    terragrunt run --all --non-interactive --working-dir {{aws_live}} -- output

# Remove everything from real AWS, including registered users
[confirm("Destroy everything in your real AWS account, including registered users? (y/N)")]
aws-destroy: _aws-check
    terragrunt run --all --non-interactive --working-dir {{aws_live}} -- destroy -auto-approve

# Draw docs/architecture.png from the services deployed to real AWS
aws-diagram: _aws-check
    uv run --quiet --script scripts/diagram.py --live {{aws_live}} --out ../docs/architecture

# Serve web/ on http://localhost:8000 against the API deployed on real AWS
web-dev: _aws-check
    #!/usr/bin/env bash
    set -euo pipefail
    out() { terragrunt run --non-interactive --log-level error --working-dir {{aws_live}}/$1 -- output -raw $2; }
    client_id="$(out auth user_pool_client_id)"
    api="$(out chat-api api_endpoint)"
    domains="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1].split(",")))' "${RAG_ALLOWED_EMAIL_DOMAINS:-uu.se,student.uu.se}")"
    printf 'window.APP_CONFIG = {"region":"%s","userPoolClientId":"%s","allowedEmailDomains":%s,"apiBase":"%s"};\n' \
      "{{aws_region}}" "$client_id" "$domains" "$api" > ../web/config.js
    echo "Wrote web/config.js; serving on http://localhost:8000"
    python3 -m http.server 8000 --bind 127.0.0.1 --directory ../web

# --- Second real AWS account: only the monthly cost budget (RAG_MANTLE_* in infra/.env) ---

# Run bootstrap, plan, apply or destroy in the second account, e.g. `just aws-mantle apply`
aws-mantle step:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{step}}" in bootstrap|plan|apply|destroy) ;; *) echo "Use bootstrap, plan, apply or destroy." >&2; exit 1 ;; esac
    for v in RAG_MANTLE_AWS_PROFILE RAG_MANTLE_AWS_ACCOUNT_ID; do
      [ -n "${!v:-}" ] || { echo "Set $v in infra/.env (see infra/.env.example)." >&2; exit 1; }
    done
    # Variables already set win over infra/.env, so _aws-check verifies the second account.
    AWS_PROFILE="$RAG_MANTLE_AWS_PROFILE" RAG_AWS_ACCOUNT_ID="$RAG_MANTLE_AWS_ACCOUNT_ID" RAG_AWS_LIVE=live/aws-mantle \
      just aws-{{step}}

# --- Trying the local stack by hand (after `just apply`) ---
# LocalStack's Cognito has no passwordless sign-in, so local users get a fixed password instead.

# Run a step of the local sign-in and chat flow (`just flow --help` lists them)
flow *args:
    @uv run --quiet --script scripts/local_flow.py {{args}}

# Page URL, API URL, user pool and client IDs of the local stack
outputs: (flow "outputs")

# Sign-in codes LocalStack has "emailed", e.g. `just codes me@student.uu.se`
codes *email: (flow "codes" email)

# Register a user on the local stack, e.g. `just signup me@student.uu.se`
signup email: (flow "signup" email)

# Sign in and send a message through CloudFront, e.g. `just chat me@student.uu.se "hello"`
chat email message:
    @uv run --quiet --script scripts/local_flow.py chat {{quote(email)}} {{quote(message)}}

# Print a snippet to paste in the browser console on the local page to sign in as email
session email: (flow "session" email)

# Start LocalStack, deploy, register email (once) and print how to chat, e.g. `just local-up me@student.uu.se`
local-up email: up phoenix-up apply
    #!/usr/bin/env bash
    set -euo pipefail
    just flow token {{quote(email)}} >/dev/null 2>&1 || just signup {{quote(email)}}
    just outputs
    echo
    echo "Chat from the terminal:  just chat {{email}} \"hello\""
    echo "Or open the page URL above and paste this in the browser console:"
    just session {{quote(email)}}

# Destroy the local environment and stop LocalStack
local-down: destroy down
