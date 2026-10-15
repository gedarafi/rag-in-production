# Your real AWS account. Values come from infra/.env (see infra/.env.example).
locals {
  target      = "aws"
  environment = "aws"
  name_prefix = "rag-aws"

  # Stockholm: Cognito here can only send email through SES in the same region.
  region = get_env("AWS_REGION", "eu-north-1")

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

  # Lets `just web-dev` (http://localhost:8000) call the API directly.
  cors_allow_origins = ["http://localhost:8000"]
}
