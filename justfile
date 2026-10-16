# Project tooling. Run `just` to list the recipes.

# Recipes run from infra/ so tenv picks up the pinned OpenTofu and Terragrunt versions.
set working-directory := "infra"
set dotenv-path := "infra/.env"

compose := "docker compose"
endpoint := env("LOCALSTACK_ENDPOINT", "http://localhost:4566")
region := env("AWS_DEFAULT_REGION", "us-east-1")
modules := "modules/s3-bucket modules/auth modules/chat-api modules/web modules/budget"
live := "live/local"
aws_live := "live/aws"
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

# Refresh the provider lock files for Linux, macOS and Windows
lock:
    for m in {{modules}}; do tofu -chdir=$m providers lock -platform=linux_amd64 -platform=linux_arm64 -platform=darwin_amd64 -platform=darwin_arm64 -platform=windows_amd64 || exit 1; done

# Unit tests with mocked providers, plus the Lambda unit tests (no LocalStack needed)
test-unit:
    for m in {{modules}}; do tofu -chdir=$m init -backend=false -input=false >/dev/null && tofu -chdir=$m test -filter=tests/unit.tftest.hcl || exit 1; done
    for d in modules/auth/lambda modules/chat-api/lambda; do python3 -m unittest discover -s $d -p 'test_*.py' || exit 1; done

# Integration test that creates and destroys a bucket on LocalStack (run `just up` first)
test-localstack:
    tofu -chdir=modules/s3-bucket init -backend=false -input=false
    tofu -chdir=modules/s3-bucket test -filter=tests/localstack.tftest.hcl

# Run lint and every test
test: lint test-unit test-localstack

# Run the same steps as CI: lint and unit tests, then LocalStack tests and e2e on a fresh LocalStack
ci: lint test-unit
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
local-up email: up apply
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
