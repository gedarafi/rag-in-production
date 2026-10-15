# Shared by live/aws/web and live/local/web; the values come from each environment's env.hcl.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/web"
}

dependency "auth" {
  config_path = "${get_terragrunt_dir()}/../auth"

  mock_outputs = {
    user_pool_client_id = "mock-client-id"
  }
  mock_outputs_allowed_terraform_commands = ["init", "validate", "plan"]
}

dependency "chat_api" {
  config_path = "${get_terragrunt_dir()}/../chat-api"

  mock_outputs = {
    api_domain = "mock.execute-api.eu-north-1.amazonaws.com"
  }
  mock_outputs_allowed_terraform_commands = ["init", "validate", "plan"]
}

inputs = {
  name_prefix           = local.env.name_prefix
  site_dir              = "${get_repo_root()}/web"
  region                = local.env.region
  user_pool_client_id   = dependency.auth.outputs.user_pool_client_id
  allowed_email_domains = local.env.allowed_email_domains
  api_domain            = dependency.chat_api.outputs.api_domain
  # LocalStack serves the API over plain HTTP on its gateway port.
  api_origin_port            = try(local.env.api_origin_port, 443)
  api_origin_protocol_policy = try(local.env.api_origin_protocol_policy, "https-only")
}
