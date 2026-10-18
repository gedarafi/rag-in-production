# Shared by live/aws/chat-api and live/local/chat-api; the values come from each environment's env.hcl.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/chat-api"

  # The Lambda's zip is made from infra/.build/chat: src/chat and src/tracing.py plus their dependencies.
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

dependency "recommender" {
  config_path = "${get_terragrunt_dir()}/../recommender"

  mock_outputs = {
    jobs_table          = "mock-chat-jobs"
    jobs_table_arn      = "arn:aws:dynamodb:us-east-1:000000000000:table/mock-chat-jobs"
    conversations_table = "mock-chat-conversations"
    queue_url           = "https://sqs.us-east-1.amazonaws.com/000000000000/mock-recommender-jobs"
    queue_arn           = "arn:aws:sqs:us-east-1:000000000000:mock-recommender-jobs"
  }
  mock_outputs_allowed_terraform_commands = ["init", "validate", "plan", "destroy"]
}

inputs = {
  name_prefix               = local.env.name_prefix
  lambda_source_dir         = "${get_repo_root()}/infra/.build/chat"
  jwt_issuer                = dependency.auth.outputs.issuer
  jwt_audience              = [dependency.auth.outputs.user_pool_client_id]
  cors_allow_origins        = local.env.cors_allow_origins
  jobs_table                = dependency.recommender.outputs.jobs_table
  jobs_table_arn            = dependency.recommender.outputs.jobs_table_arn
  conversations_table       = dependency.recommender.outputs.conversations_table
  queue_url                 = dependency.recommender.outputs.queue_url
  queue_arn                 = dependency.recommender.outputs.queue_arn
  tracing_endpoint          = local.env.tracing_endpoint
  tracing_project           = local.env.tracing_project
  tracing_space_id          = local.env.tracing_space_id
  tracing_api_key_parameter = local.env.tracing_api_key_parameter
}
