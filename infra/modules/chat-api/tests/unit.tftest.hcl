# Plan-only tests with mocked providers. They need no LocalStack and no credentials.

mock_provider "aws" {
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/rag-unit-chat"
    }
  }
  mock_resource "aws_lambda_function" {
    defaults = {
      arn        = "arn:aws:lambda:eu-north-1:123456789012:function:rag-unit-chat"
      invoke_arn = "arn:aws:apigateway:eu-north-1:lambda:path/2015-03-31/functions/arn:aws:lambda:eu-north-1:123456789012:function:rag-unit-chat/invocations"
    }
  }
  mock_resource "aws_apigatewayv2_api" {
    defaults = {
      api_endpoint  = "https://abc123.execute-api.eu-north-1.amazonaws.com"
      execution_arn = "arn:aws:execute-api:eu-north-1:123456789012:abc123"
    }
  }
  mock_resource "aws_cloudwatch_log_group" {
    defaults = {
      arn = "arn:aws:logs:eu-north-1:123456789012:log-group:test"
    }
  }
}

mock_provider "archive" {}

variables {
  name_prefix  = "rag-unit"
  jwt_issuer   = "https://cognito-idp.eu-north-1.amazonaws.com/eu-north-1_TEST"
  jwt_audience = ["client123"]
  model_id     = "us.amazon.nova-micro-v1:0"
}

run "protects_the_chat_route_with_cognito" {
  command = plan

  assert {
    condition     = aws_apigatewayv2_route.chat.route_key == "POST /api/chat"
    error_message = "The chat route must be POST /api/chat, matching the CloudFront /api/* path."
  }

  assert {
    condition     = aws_apigatewayv2_route.chat.authorization_type == "JWT"
    error_message = "The chat route must require a Cognito token."
  }

  assert {
    condition     = aws_apigatewayv2_authorizer.cognito.jwt_configuration[0].issuer == var.jwt_issuer
    error_message = "The authorizer must trust the user pool issuer."
  }

  assert {
    condition     = aws_apigatewayv2_authorizer.cognito.jwt_configuration[0].audience == toset(["client123"])
    error_message = "The authorizer must only accept tokens for the web client."
  }
}

run "throttles_by_default" {
  command = plan

  assert {
    condition     = aws_apigatewayv2_stage.default.default_route_settings[0].throttling_rate_limit == 5
    error_message = "The API must be throttled to limit costs."
  }
}

run "no_cors_unless_asked" {
  command = plan

  assert {
    condition     = length(aws_apigatewayv2_api.this.cors_configuration) == 0
    error_message = "CORS must be off by default; the page calls the API on its own origin."
  }
}

run "outputs_the_api_domain" {
  command = apply

  assert {
    condition     = output.api_domain == "abc123.execute-api.eu-north-1.amazonaws.com"
    error_message = "api_domain must be the host name without https://."
  }
}

run "rejects_a_non_https_issuer" {
  command = plan

  variables {
    jwt_issuer = "http://example.com"
  }

  expect_failures = [var.jwt_issuer]
}

run "outputs_a_bare_host_for_a_localstack_endpoint" {
  command = apply

  override_resource {
    target = aws_apigatewayv2_api.this
    values = {
      api_endpoint  = "http://abc123.execute-api.localhost.localstack.cloud:4566"
      execution_arn = "arn:aws:execute-api:us-east-1:000000000000:abc123"
    }
  }

  assert {
    condition     = output.api_domain == "abc123.execute-api.localhost.localstack.cloud"
    error_message = "api_domain must drop the scheme and the port."
  }
}

run "configures_the_model" {
  command = plan

  assert {
    condition     = aws_lambda_function.chat.environment[0].variables.MODEL_ID == "us.amazon.nova-micro-v1:0"
    error_message = "The Lambda must call the configured model."
  }

  assert {
    condition     = aws_lambda_function.chat.environment[0].variables.MAX_TOKENS == "512"
    error_message = "Answers must be capped at max_tokens."
  }

  assert {
    condition     = aws_lambda_function.chat.timeout < 30
    error_message = "The Lambda must finish before API Gateway's 30-second limit."
  }
}

run "may_invoke_only_the_configured_model" {
  command = apply

  assert {
    condition = jsondecode(aws_iam_role_policy.chat_bedrock.policy).Statement[0].Resource == [
      "arn:aws:bedrock:*::foundation-model/amazon.nova-micro-v1:0",
      "arn:aws:bedrock:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:inference-profile/us.amazon.nova-micro-v1:0",
    ]
    error_message = "The Lambda may invoke the inference profile and its foundation model, nothing else."
  }
}

run "allows_a_plain_foundation_model" {
  command = apply

  variables {
    model_id = "amazon.nova-lite-v1:0"
  }

  assert {
    condition     = contains(jsondecode(aws_iam_role_policy.chat_bedrock.policy).Statement[0].Resource, "arn:aws:bedrock:*::foundation-model/amazon.nova-lite-v1:0")
    error_message = "A model ID without a geography prefix is the foundation model itself."
  }
}

run "rejects_an_empty_model" {
  command = plan

  variables {
    model_id = " "
  }

  expect_failures = [var.model_id]
}
