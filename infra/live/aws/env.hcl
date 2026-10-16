# Your real AWS account. Values come from infra/.env (see infra/.env.example).
locals {
  target      = "aws"
  environment = "aws"
  name_prefix = "rag-aws"

  # Stockholm: Cognito here can only send email through SES in the same region.
  # region = get_env("AWS_REGION", "eu-north-1")
  region = get_env("AWS_REGION", "us-east-2")


  # The provider refuses to touch any other account, so a wrong AWS_PROFILE can't deploy elsewhere.
  # Defaults are empty so `just lint` works without credentials; the aws-* recipes require them.
  account_id   = get_env("RAG_AWS_ACCOUNT_ID", "")
  state_bucket = "rag-in-production-tfstate-${get_env("RAG_AWS_ACCOUNT_ID", "")}"

  # Sign-in settings.
  allowed_email_domains = split(",", get_env("RAG_ALLOWED_EMAIL_DOMAINS", "uu.se,student.uu.se"))
  ses_identity          = get_env("RAG_SES_IDENTITY", "")
  route53_zone_id       = get_env("RAG_ROUTE53_ZONE_ID", "") == "" ? null : get_env("RAG_ROUTE53_ZONE_ID")
  create_ses_identity   = true

  # Switch to ACTIVE once real users depend on the pool.
  deletion_protection = "INACTIVE"

  # Chat model: a Bedrock model or cross-region inference profile available in the region.
  model_id   = get_env("RAG_BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")
  max_tokens = 512

  # Monthly cost budget for the whole account: emails at 80 % actual and 100 % forecasted spend.
  # The emails default to RAG_SES_IDENTITY when that is an address (a domain is not); set RAG_BUDGET_EMAILS otherwise.
  budget_limit_usd = tonumber(get_env("RAG_BUDGET_LIMIT_USD", "10"))
  budget_emails    = compact(split(",", get_env("RAG_BUDGET_EMAILS", strcontains(get_env("RAG_SES_IDENTITY", ""), "@") ? get_env("RAG_SES_IDENTITY", "") : "")))

  # Lets `just web-dev` (http://localhost:8000) call the API directly.
  cors_allow_origins = ["http://localhost:8000"]
}
