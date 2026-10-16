# Project tooling. Run `just` to list the recipes.

# Recipes run from infra/ so tenv picks up the pinned OpenTofu and Terragrunt versions.
set working-directory := "infra"
set dotenv-path := "infra/.env"

compose := "docker compose"
endpoint := env("LOCALSTACK_ENDPOINT", "http://localhost:4566")
region := env("AWS_DEFAULT_REGION", "us-east-1")
modules := "modules/s3-bucket modules/auth modules/chat-api modules/web"
live := "live/local"

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

# --- Trying the local stack by hand (after `just apply`) ---
# LocalStack's Cognito has no passwordless sign-in, so local users get a fixed password instead.

# Run a step of the local sign-in and chat flow (`just flow help` lists them)
flow *args:
    @python3 -I scripts/local_flow.py {{args}}

# Page URL, API URL, user pool and client IDs of the local stack
outputs: (flow "outputs")

# Sign-in codes LocalStack has "emailed", e.g. `just codes me@student.uu.se`
codes *email: (flow "codes" email)

# Register a user on the local stack, e.g. `just signup me@student.uu.se`
signup email: (flow "signup" email)

# Sign in and send a message through CloudFront, e.g. `just chat me@student.uu.se "hello"`
chat email message:
    @python3 -I scripts/local_flow.py chat {{quote(email)}} {{quote(message)}}

# Print a snippet to paste in the browser console on the local page to sign in as email
session email: (flow "session" email)
