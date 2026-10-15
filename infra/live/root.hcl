# Shared Terragrunt settings for every unit under infra/live/.
# The local environment targets LocalStack; set LOCALSTACK_ENDPOINT to point somewhere else.

locals {
  env                 = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
  localstack_endpoint = get_env("LOCALSTACK_ENDPOINT", "http://localhost:4566")
}

# State stays on disk next to this file (gitignored). A real AWS environment would use an S3 backend.
remote_state {
  backend = "local"
  config = {
    path = "${get_parent_terragrunt_dir()}/.state/${path_relative_to_include()}/terraform.tfstate"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<-EOF
    provider "aws" {
      region                      = "${local.env.region}"
      access_key                  = "test"
      secret_key                  = "test"
      s3_use_path_style           = true
      skip_credentials_validation = true
      skip_metadata_api_check     = true
      skip_requesting_account_id  = true

      endpoints {
        s3  = "${local.localstack_endpoint}"
        sts = "${local.localstack_endpoint}"
      }

      default_tags {
        tags = {
          project     = "rag-in-production"
          environment = "${local.env.environment}"
          managed_by  = "terragrunt"
        }
      }
    }
  EOF
}
