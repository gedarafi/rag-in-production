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

  # Recommender model. "openai" calls Chat Completions on bedrock-mantle (bedrock-mantle.<region>.api.aws),
  # whose quotas are separate from bedrock-runtime's; the requests show up under the default project
  # in the console. Qwen3 32B follows the pipeline's JSON schemas strictly there (see INFRA.md).
  # "converse" calls Converse on bedrock-runtime with a model or inference profile such as
  # us.amazon.nova-micro-v1:0.
  model_id               = get_env("RAG_BEDROCK_MODEL_ID", "qwen.qwen3-32b")
  model_api              = get_env("RAG_BEDROCK_MODEL_API", "openai")
  worker_timeout_seconds = 120
  model_timeout_seconds  = 60

  # Artifact bundles (`just aws-recommender-publish`). Bucket names are global, hence the account ID.
  artifact_bucket_name   = "rag-aws-recommender-artifacts-${get_env("RAG_AWS_ACCOUNT_ID", "")}"
  force_destroy          = false
  point_in_time_recovery = true

  # Traces go to Arize AX (a Lambda in AWS can't reach the Phoenix on a developer's machine), but only
  # once RAG_ARIZE_SPACE_ID is set. The API key stays in SSM: store it with `just aws-arize-key`.
  arize_space_id            = get_env("RAG_ARIZE_SPACE_ID", "")
  tracing_endpoint          = local.arize_space_id == "" ? "" : get_env("RAG_TRACING_ENDPOINT", "https://otlp.arize.com/v1/traces")
  tracing_project           = get_env("RAG_TRACING_PROJECT", "rag-aws")
  tracing_space_id          = local.arize_space_id
  tracing_api_key_parameter = local.arize_space_id == "" ? "" : "/${local.name_prefix}/arize-api-key"

  # Monthly cost budget for the whole account: emails at 80 % actual and 100 % forecasted spend.
  # The emails default to RAG_SES_IDENTITY when that is an address (a domain is not); set RAG_BUDGET_EMAILS otherwise.
  budget_limit_usd = tonumber(get_env("RAG_BUDGET_LIMIT_USD", "10"))
  budget_emails    = compact(split(",", get_env("RAG_BUDGET_EMAILS", strcontains(get_env("RAG_SES_IDENTITY", ""), "@") ? get_env("RAG_SES_IDENTITY", "") : "")))

  # The recommender's alarms (dead letters, errors, slow answers) go to the same addresses.
  alarm_emails = local.budget_emails

  # Lets `just web-dev` (http://localhost:8000) call the API directly.
  cors_allow_origins = ["http://localhost:8000"]
}
