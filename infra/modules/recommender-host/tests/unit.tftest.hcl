# Plan-only tests with a mocked provider. They need no LocalStack and no credentials.

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }
  mock_data "aws_region" {
    defaults = {
      region = "us-east-2"
    }
  }
  mock_data "aws_ec2_managed_prefix_list" {
    defaults = {
      id = "pl-cloudfront"
    }
  }
  mock_data "aws_ssm_parameter" {
    defaults = {
      value = "ami-0arm64"
    }
  }
  mock_resource "aws_s3_bucket" {
    defaults = {
      arn = "arn:aws:s3:::rag-unit-recommender-123456789012"
    }
  }
}

variables {
  name_prefix       = "rag-unit"
  cognito_issuer    = "https://cognito-idp.us-east-2.amazonaws.com/us-east-2_pool"
  cognito_client_id = "client123"
}

run "only_cloudfront_reaches_the_service" {
  command = plan

  assert {
    condition     = contains(one(aws_security_group.host.ingress).prefix_list_ids, "pl-cloudfront") && try(length(one(aws_security_group.host.ingress).cidr_blocks), 0) == 0
    error_message = "Ingress must come only from CloudFront's origin-facing prefix list."
  }

  assert {
    condition     = one(aws_security_group.host.ingress).from_port == 8080 && one(aws_security_group.host.ingress).to_port == 8080
    error_message = "Only the service port may be open."
  }
}

run "instance_is_small_graviton_with_imdsv2" {
  command = plan

  assert {
    condition     = aws_instance.host[0].instance_type == "t4g.medium" && aws_instance.host[0].ami == "ami-0arm64"
    error_message = "Default to t4g.medium on the arm64 Amazon Linux AMI."
  }

  assert {
    condition     = aws_instance.host[0].metadata_options[0].http_tokens == "required"
    error_message = "IMDSv2 must be required."
  }

  assert {
    condition     = aws_instance.host[0].root_block_device[0].encrypted
    error_message = "The root volume must be encrypted."
  }
}

run "user_data_configures_bedrock_and_cognito" {
  command = plan

  assert {
    condition = alltrue([for line in [
      "RECOMMENDER_BUCKET=rag-unit-recommender-123456789012",
      "COGNITO_ISSUER=https://cognito-idp.us-east-2.amazonaws.com/us-east-2_pool",
      "COGNITO_CLIENT_ID=client123",
      "RAG_PROVIDER=bedrock",
      "RAG_BEDROCK_MODEL=qwen.qwen3-32b",
      "recommender serve --host 0.0.0.0 --port 8080",
    ] : strcontains(aws_instance.host[0].user_data, line)])
    error_message = "The instance must be configured for Bedrock, the data bucket and the user pool."
  }
}

run "role_reads_only_its_bucket_and_calls_mantle" {
  command = plan

  assert {
    condition = jsondecode(aws_iam_role_policy.host.policy).Statement[0].Resource == [
      "arn:aws:s3:::rag-unit-recommender-123456789012/recommender/*",
      "arn:aws:s3:::rag-unit-recommender-123456789012/releases/*",
    ]
    error_message = "The instance may read only the library and the wheel."
  }

  assert {
    condition     = jsondecode(aws_iam_role_policy.host.policy).Statement[2].Action == "bedrock-mantle:CreateInference"
    error_message = "The instance must be allowed to call bedrock-mantle."
  }
}

run "without_instance_only_bucket_role_and_network" {
  command = plan

  variables {
    create_instance = false
  }

  assert {
    condition     = length(aws_instance.host) == 0 && length(aws_eip.host) == 0 && output.origin_domain == ""
    error_message = "create_instance = false must not run an instance."
  }
}
