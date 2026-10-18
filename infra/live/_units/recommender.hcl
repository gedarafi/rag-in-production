# Shared by live/aws/recommender and live/local/recommender; the values come from each environment's env.hcl.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules//recommender"

  # Build the worker image and push it as "current"; the module deploys that image's digest.
  before_hook "build_recommender_image" {
    commands = ["plan", "apply"]
    # Hooks can't read dependency outputs; the script looks the repository up by its name instead.
    execute = ["${get_repo_root()}/infra/scripts/build_recommender_image.sh", "${local.env.name_prefix}-recommender", local.env.region, local.env.target]
  }
}

dependency "registry" {
  config_path = "${get_terragrunt_dir()}/../recommender-registry"

  mock_outputs = {
    repository_name = "mock-recommender"
    repository_url  = "000000000000.dkr.ecr.us-east-1.amazonaws.com/mock-recommender"
  }
  mock_outputs_allowed_terraform_commands = ["init", "validate", "plan", "destroy"]
}

dependency "vector_db" {
  config_path = "${get_terragrunt_dir()}/../vector-db"

  mock_outputs = {
    cluster_arn = "arn:aws:rds:us-east-1:000000000000:cluster:mock-vectors"
    secret_arn  = "arn:aws:secretsmanager:us-east-1:000000000000:secret:mock-vectors"
    database    = "vectors"
  }
  mock_outputs_allowed_terraform_commands = ["init", "validate", "plan", "destroy"]
}

inputs = {
  name_prefix               = local.env.name_prefix
  repository_name           = dependency.registry.outputs.repository_name
  repository_url            = dependency.registry.outputs.repository_url
  artifact_bucket_name      = local.env.artifact_bucket_name
  vector_db                 = dependency.vector_db.outputs
  model_id                  = local.env.model_id
  model_api                 = local.env.model_api
  timeout_seconds           = local.env.worker_timeout_seconds
  model_timeout_seconds     = local.env.model_timeout_seconds
  ollama_url                = try(local.env.ollama_url, "")
  point_in_time_recovery    = local.env.point_in_time_recovery
  alarm_emails              = local.env.alarm_emails
  force_destroy             = local.env.force_destroy
  tracing_endpoint          = local.env.tracing_endpoint
  tracing_project           = local.env.tracing_project
  tracing_space_id          = local.env.tracing_space_id
  tracing_api_key_parameter = local.env.tracing_api_key_parameter
}
