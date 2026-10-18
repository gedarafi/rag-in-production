# Shared Terragrunt settings for every unit under infra/live/.
# Each environment's env.hcl sets `target`:
#   "localstack" (live/local): emulated AWS, state on disk next to this file.
#   "aws"        (live/aws):   your real account, state in an S3 bucket (`just aws-bootstrap`).
#                (live/aws-mantle): a second real account with only the budget (`just aws-mantle bootstrap`).

locals {
  env                 = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
  is_localstack       = local.env.target == "localstack"
  localstack_endpoint = get_env("LOCALSTACK_ENDPOINT", "http://localhost:4566")
  state_key           = "${path_relative_to_include()}/terraform.tfstate"

  default_tags = <<-EOF
    default_tags {
      tags = {
        project     = "rag-in-production"
        environment = "${local.env.environment}"
        managed_by  = "terragrunt"
      }
    }
  EOF
}

generate "backend" {
  path      = "backend.tf"
  if_exists = "overwrite_terragrunt"
  contents = local.is_localstack ? (<<-EOF
    terraform {
      backend "local" {
        path = "${get_parent_terragrunt_dir()}/.state/${local.state_key}"
      }
    }
  EOF
    ) : (<<-EOF
    terraform {
      backend "s3" {
        bucket       = "${try(local.env.state_bucket, "")}"
        key          = "${local.state_key}"
        region       = "${local.env.region}"
        encrypt      = true
        use_lockfile = true
      }
    }
  EOF
  )
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents = local.is_localstack ? (<<-EOF
    provider "aws" {
      region                      = "${local.env.region}"
      access_key                  = "test"
      secret_key                  = "test"
      s3_use_path_style           = true
      skip_credentials_validation = true
      skip_metadata_api_check     = true
      skip_requesting_account_id  = true

      endpoints {
        apigatewayv2   = "${local.localstack_endpoint}"
        cloudfront     = "${local.localstack_endpoint}"
        cloudwatch     = "${local.localstack_endpoint}"
        cognitoidp     = "${local.localstack_endpoint}"
        dynamodb       = "${local.localstack_endpoint}"
        ec2            = "${local.localstack_endpoint}"
        ecr            = "${local.localstack_endpoint}"
        iam            = "${local.localstack_endpoint}"
        lambda         = "${local.localstack_endpoint}"
        logs           = "${local.localstack_endpoint}"
        rds            = "${local.localstack_endpoint}"
        route53        = "${local.localstack_endpoint}"
        s3             = "${local.localstack_endpoint}"
        secretsmanager = "${local.localstack_endpoint}"
        sesv2          = "${local.localstack_endpoint}"
        sns            = "${local.localstack_endpoint}"
        sqs            = "${local.localstack_endpoint}"
        ssm            = "${local.localstack_endpoint}"
        sts            = "${local.localstack_endpoint}"
      }

    ${local.default_tags}
    }
  EOF
    ) : (<<-EOF
    # Credentials come from your AWS CLI login (AWS_PROFILE in infra/.env, e.g. after `aws sso login`).
    provider "aws" {
      region              = "${local.env.region}"
      allowed_account_ids = ["${try(local.env.account_id, "")}"]

    ${local.default_tags}
    }
  EOF
  )
}
