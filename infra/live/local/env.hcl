locals {
  target      = "localstack"
  environment = "local"
  name_prefix = "rag-local"
  region      = "us-east-1"

  # Sign-in settings. LocalStack accepts any sender without verification and sends no real email;
  # the codes appear in its logs (`just codes`).
  allowed_email_domains = ["uu.se", "student.uu.se"]
  ses_identity          = "no-reply@example.com"
  create_ses_identity   = false # LocalStack can't tag SESv2 identities
  route53_zone_id       = null
  deletion_protection   = "INACTIVE"
  cors_allow_origins    = ["http://localhost:8000"]

  # LocalStack serves the HTTP API on its gateway port, over plain HTTP.
  api_origin_port            = 4566
  api_origin_protocol_policy = "http-only"
}
