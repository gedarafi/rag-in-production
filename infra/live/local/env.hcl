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

  # Recommender model. By default the ollama service of docker-compose.yml (RAG_LOCAL_MODEL, on the GPU
  # when Docker has the NVIDIA runtime), called with schema-constrained output, which a small local
  # model needs and LocalStack's Bedrock emulation can't pass on. CI sets RAG_LOCAL_MODEL_API=stub: a
  # deterministic stand-in (src/recommender/stub.py) that checks the whole path in seconds and says
  # nothing about answer quality. RAG_LOCAL_MODEL_API=converse uses LocalStack's Bedrock instead.
  model_api = get_env("RAG_LOCAL_MODEL_API", "ollama")
  model_id = {
    ollama   = get_env("RAG_LOCAL_MODEL", "llama3.2:3b")
    converse = "ollama.${get_env("LOCALSTACK_BEDROCK_MODEL", "llama3.2:3b")}"
    stub     = "stub"
  }[local.model_api]
  ollama_url             = "http://ollama:11434"
  worker_timeout_seconds = local.model_api == "stub" ? 120 : 900
  model_timeout_seconds  = 300

  # Artifact bundles (`just recommender-publish`), and what may be thrown away with the stack.
  artifact_bucket_name   = "rag-local-recommender-artifacts"
  force_destroy          = true
  point_in_time_recovery = false
  alarm_emails           = []

  # The Lambdas' traces go to Phoenix in docker-compose.yml (`just phoenix-up`, UI on :6006). The Lambda
  # containers join the Compose network (LAMBDA_DOCKER_NETWORK), where Phoenix is reachable as phoenix.
  tracing_endpoint          = "http://phoenix:6006/v1/traces"
  tracing_project           = "rag-in-production"
  tracing_space_id          = "" # Phoenix needs no space or API key
  tracing_api_key_parameter = ""

  # LocalStack serves the HTTP API on its gateway port, over plain HTTP.
  api_origin_port            = 4566
  api_origin_protocol_policy = "http-only"
}
