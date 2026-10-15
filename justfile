# Project tooling. Run `just` to list the recipes.

# Recipes run from infra/ so tenv picks up the pinned OpenTofu and Terragrunt versions.
set working-directory := "infra"
set dotenv-path := "infra/.env"

compose := "docker compose"
endpoint := env("LOCALSTACK_ENDPOINT", "http://localhost:4566")
region := env("AWS_DEFAULT_REGION", "us-east-1")
module := "modules/s3-bucket"
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

# Follow the LocalStack logs (`just logs --no-follow` to print and exit)
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

# Check formatting and validate the module and Terragrunt config
lint:
    tofu fmt -check -recursive
    terragrunt hcl fmt --check --working-dir live
    tofu -chdir={{module}} init -backend=false -input=false
    tofu -chdir={{module}} validate
    terragrunt hcl validate --working-dir live

# Refresh the provider lock file for Linux, macOS and Windows
lock:
    tofu -chdir={{module}} providers lock -platform=linux_amd64 -platform=linux_arm64 -platform=darwin_amd64 -platform=darwin_arm64 -platform=windows_amd64

# Unit tests with a mocked provider (no LocalStack needed)
test-unit:
    tofu -chdir={{module}} init -backend=false -input=false
    tofu -chdir={{module}} test -filter=tests/unit.tftest.hcl

# Integration test that creates and destroys a bucket on LocalStack (run `just up` first)
test-localstack:
    tofu -chdir={{module}} init -backend=false -input=false
    tofu -chdir={{module}} test -filter=tests/localstack.tftest.hcl

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

# Apply the local environment, check the bucket with the AWS CLI, then destroy it
e2e:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'just destroy' EXIT
    just apply
    just aws s3api head-bucket --bucket rag-documents-local
    status="$(just aws s3api get-bucket-versioning --bucket rag-documents-local --query Status --output text)"
    [ "$status" = "Enabled" ] || { echo "expected versioning Enabled, got '$status'" >&2; exit 1; }
    echo "e2e: bucket rag-documents-local exists with versioning enabled"
