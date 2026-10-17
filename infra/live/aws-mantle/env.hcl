# A second real AWS account that runs only the monthly cost budget (`just aws-mantle ...`), for
# calling Bedrock Mantle by hand. Values come from infra/.env (see infra/.env.example).
locals {
  aws = read_terragrunt_config("${get_repo_root()}/infra/live/aws/env.hcl").locals

  target      = "aws"
  environment = "aws-mantle"
  name_prefix = "rag-mantle"
  region      = local.aws.region

  # The provider refuses to touch any other account, so this never reaches the chatbot's account.
  account_id   = get_env("RAG_MANTLE_AWS_ACCOUNT_ID", "")
  state_bucket = "rag-in-production-tfstate-${get_env("RAG_MANTLE_AWS_ACCOUNT_ID", "")}"

  # Same limit and emails as live/aws (RAG_BUDGET_LIMIT_USD, RAG_BUDGET_EMAILS).
  budget_limit_usd = local.aws.budget_limit_usd
  budget_emails    = local.aws.budget_emails
}
