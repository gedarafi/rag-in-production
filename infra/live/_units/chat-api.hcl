# Shared by live/aws/chat-api and live/local/chat-api; the values come from each environment's env.hcl.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/chat-api"

  # The Lambda's zip is made from infra/.build/chat: src/chat/handler.py plus its dependencies.
  before_hook "build_chat_lambda" {
    commands = ["plan", "apply", "destroy"]
    execute  = ["${get_repo_root()}/infra/scripts/build_chat_lambda.sh"]
  }
}

dependency "auth" {
  config_path = "${get_terragrunt_dir()}/../auth"

  mock_outputs = {
    issuer              = "https://cognito-idp.eu-north-1.amazonaws.com/eu-north-1_MOCK"
    user_pool_client_id = "mock-client-id"
  }
  mock_outputs_allowed_terraform_commands = ["init", "validate", "plan"]
}

inputs = {
  name_prefix        = local.env.name_prefix
  lambda_source_dir  = "${get_repo_root()}/infra/.build/chat"
  jwt_issuer         = dependency.auth.outputs.issuer
  jwt_audience       = [dependency.auth.outputs.user_pool_client_id]
  cors_allow_origins = local.env.cors_allow_origins
  model_id           = local.env.model_id
  model_api          = local.env.model_api
  max_tokens         = local.env.max_tokens
  tracing_endpoint   = local.env.tracing_endpoint
}
