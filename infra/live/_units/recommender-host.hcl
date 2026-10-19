# Shared by live/aws/recommender-host and live/local/recommender-host; the values come from each
# environment's env.hcl.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules//recommender-host"
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
  name_prefix         = local.env.name_prefix
  cognito_issuer      = dependency.auth.outputs.issuer
  cognito_client_id   = dependency.auth.outputs.user_pool_client_id
  create_instance     = local.env.recommender_instance
  instance_type       = local.env.recommender_instance_type
  bedrock_model       = local.env.recommender_model_id
  ingress_cidr_blocks = try(local.env.recommender_ingress_cidr_blocks, [])
  force_destroy       = local.env.target == "localstack"
  # Same traces as the chat Lambda: Arize AX on AWS (the local service sets its own Phoenix endpoint).
  tracing_endpoint          = local.env.tracing_endpoint
  tracing_project           = local.env.tracing_project
  tracing_space_id          = local.env.tracing_space_id
  tracing_api_key_parameter = local.env.tracing_api_key_parameter
}
