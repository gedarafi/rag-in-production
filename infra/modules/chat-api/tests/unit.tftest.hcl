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
  name_prefix         = "rag-unit"
  lambda_source_dir   = "../../../src/chat"
  jwt_issuer          = "https://cognito-idp.eu-north-1.amazonaws.com/eu-north-1_TEST"
  jwt_audience        = ["client123"]
  jobs_table          = "rag-unit-chat-jobs"
  jobs_table_arn      = "arn:aws:dynamodb:eu-north-1:123456789012:table/rag-unit-chat-jobs"
  conversations_table = "rag-unit-chat-conversations"
  queue_url           = "https://sqs.eu-north-1.amazonaws.com/123456789012/rag-unit-recommender-jobs"
  queue_arn           = "arn:aws:sqs:eu-north-1:123456789012:rag-unit-recommender-jobs"
}

run "protects_the_chat_route_with_cognito" {
  command = plan

  assert {
    condition     = aws_apigatewayv2_route.chat.route_key == "POST /api/chat"
    error_message = "The chat route must be POST /api/chat, matching the CloudFront /api/* path."
  }

  assert {
    condition     = aws_apigatewayv2_route.chat.authorization_type == "JWT" && aws_apigatewayv2_route.job.authorization_type == "JWT"
    error_message = "Both chat routes must require a Cognito token."
  }

  assert {
    condition     = aws_apigatewayv2_route.job.route_key == "GET /api/chat/{job_id}"
    error_message = "The page polls GET /api/chat/{job_id} for the answer."
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

run "queues_questions_for_the_worker" {
  command = plan

  assert {
    condition = {
      for k, v in aws_lambda_function.chat.environment[0].variables : k => v if !startswith(k, "TRACING_")
      } == {
      JOBS_TABLE          = "rag-unit-chat-jobs"
      CONVERSATIONS_TABLE = "rag-unit-chat-conversations"
      QUEUE_URL           = "https://sqs.eu-north-1.amazonaws.com/123456789012/rag-unit-recommender-jobs"
    }
    error_message = "The Lambda must know the jobs table and the queue, and nothing about the model."
  }

  assert {
    condition     = aws_lambda_function.chat.environment[0].variables.TRACING_ENDPOINT == ""
    error_message = "Tracing must be off unless a tracing_endpoint is given."
  }

  assert {
    condition     = length(aws_iam_role_policy.chat_tracing_key) == 0
    error_message = "The Lambda must not read SSM unless a tracing API key parameter is given."
  }

  assert {
    condition     = aws_lambda_function.chat.timeout < 30
    error_message = "The Lambda must finish before API Gateway's 30-second limit."
  }
}

run "may_only_queue_and_read_jobs" {
  command = apply

  assert {
    condition = jsondecode(aws_iam_role_policy.chat_jobs.policy).Statement == [
      {
        Sid      = "Jobs"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem"]
        Resource = "arn:aws:dynamodb:eu-north-1:123456789012:table/rag-unit-chat-jobs"
      },
      {
        Sid      = "Queue"
        Effect   = "Allow"
        Action   = "sqs:SendMessage"
        Resource = "arn:aws:sqs:eu-north-1:123456789012:rag-unit-recommender-jobs"
      },
    ]
    error_message = "The Lambda may only create and read jobs and send them to the queue."
  }
}

run "traces_to_arize_with_the_key_from_ssm" {
  command = apply

  variables {
    tracing_endpoint          = "https://otlp.arize.com/v1/traces"
    tracing_project           = "rag-aws"
    tracing_space_id          = "space-123"
    tracing_api_key_parameter = "/rag-aws/arize-api-key"
  }

  assert {
    condition = {
      for k, v in aws_lambda_function.chat.environment[0].variables : k => v if startswith(k, "TRACING_")
      } == {
      TRACING_ENDPOINT          = "https://otlp.arize.com/v1/traces"
      TRACING_PROJECT           = "rag-aws"
      TRACING_SPACE_ID          = "space-123"
      TRACING_API_KEY_PARAMETER = "/rag-aws/arize-api-key"
    }
    error_message = "The Lambda must get the endpoint, project, space and the key's parameter name (never the key)."
  }

  assert {
    condition = jsondecode(aws_iam_role_policy.chat_tracing_key[0].policy).Statement == [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = "arn:aws:ssm:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:parameter/rag-aws/arize-api-key"
    }]
    error_message = "The Lambda may read only the tracing key's parameter."
  }
}

run "rejects_a_relative_key_parameter" {
  command = plan

  variables {
    tracing_api_key_parameter = "rag-aws/arize-api-key"
  }

  expect_failures = [var.tracing_api_key_parameter]
}
