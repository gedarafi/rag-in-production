locals {
  target      = "localstack"
  environment = "local"
  name_prefix = "rag-local"
  region      = "us-east-1"

  # Sign-in settings. LocalStack sends no real email; the codes land in its SES inbox (`just codes`).
  # It does require a verified sender: scripts/local_flow.py verifies ses_identity before signing up.
  allowed_email_domains = ["uu.se", "student.uu.se"]
  ses_identity          = "no-reply@example.com"
  create_ses_identity   = false # LocalStack can't tag SESv2 identities
  route53_zone_id       = null
  deletion_protection   = "INACTIVE"
  cors_allow_origins    = ["http://localhost:8000"]

  # Chat model. LocalStack runs ollama.<model> on CPU (see docker-compose.yml); a tiny model keeps
  # CI fast, and short answers stay well inside API Gateway's 30-second limit. LocalStack emulates
  # Converse on bedrock-runtime, not bedrock-mantle.
  model_id   = "ollama.qwen2.5:0.5b"
  model_api  = "converse"
  max_tokens = 128

  # The Lambda's traces go to Phoenix in docker-compose.yml (`just phoenix-up`, UI on :6006). The Lambda
  # containers join the Compose network (LAMBDA_DOCKER_NETWORK), where Phoenix is reachable as phoenix.
  tracing_endpoint          = "http://phoenix:6006/v1/traces"
  tracing_project           = "rag-in-production"
  tracing_space_id          = "" # Phoenix needs no space or API key
  tracing_api_key_parameter = ""

  # LocalStack serves the HTTP API on its gateway port, over plain HTTP.
  api_origin_port            = 4566
  api_origin_protocol_policy = "http-only"
}
