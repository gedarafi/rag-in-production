# HTTP API with a Cognito JWT authorizer in front of the chat Lambda.
# The Lambda answers with a Bedrock model, through Converse on bedrock-runtime or OpenAI-compatible
# Chat Completions on bedrock-mantle (var.model_api); retrieval from the course documents comes later.

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  function_name = "${var.name_prefix}-chat"

  # A cross-region inference profile (us.amazon.nova-micro-v1:0) routes to the foundation model
  # (amazon.nova-micro-v1:0) in any region of its geography; IAM must allow both.
  base_model_id = try(regex("^(?:us|eu|apac|jp|au|ca|us-gov|global)\\.(.+)$", var.model_id)[0], var.model_id)

  # Converse authorizes the model. bedrock-mantle authorizes the project the request runs in: the
  # account's default project, since the Lambda names none.
  bedrock_statement = {
    converse = {
      Effect = "Allow"
      Action = "bedrock:InvokeModel" # also covers the Converse API
      Resource = [
        "arn:aws:bedrock:*::foundation-model/${local.base_model_id}",
        "arn:aws:bedrock:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.model_id}",
      ]
    }
    openai = {
      Effect   = "Allow"
      Action   = "bedrock-mantle:CreateInference"
      Resource = ["arn:aws:bedrock-mantle:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:project/default"]
    }
  }[var.model_api]
}

# ---------- Chat Lambda ----------

data "archive_file" "chat" {
  type        = "zip"
  source_dir  = var.lambda_source_dir
  output_path = "${path.module}/build/chat.zip"
}

resource "aws_iam_role" "chat" {
  name = local.function_name
  tags = var.tags

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "chat_logs" {
  role       = aws_iam_role.chat.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "chat_bedrock" {
  name = "invoke-model"
  role = aws_iam_role.chat.id

  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [local.bedrock_statement]
  })
}

# The tracing API key (Arize AX) stays in SSM, out of the Lambda's environment and the state; the
# Lambda reads it at cold start. The default aws/ssm key needs no KMS grant.
resource "aws_iam_role_policy" "chat_tracing_key" {
  count = var.tracing_api_key_parameter == "" ? 0 : 1
  name  = "read-tracing-key"
  role  = aws_iam_role.chat.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = "arn:aws:ssm:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:parameter/${trimprefix(var.tracing_api_key_parameter, "/")}"
    }]
  })
}

resource "aws_cloudwatch_log_group" "chat" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_lambda_function" "chat" {
  function_name    = local.function_name
  role             = aws_iam_role.chat.arn
  runtime          = "python3.13"
  handler          = "handler.handler"
  filename         = data.archive_file.chat.output_path
  source_code_hash = data.archive_file.chat.output_base64sha256
  timeout          = 29 # API Gateway gives up after 30 seconds
  memory_size      = 256
  tags             = var.tags

  environment {
    variables = {
      MODEL_ID                  = var.model_id
      MODEL_API                 = var.model_api
      MAX_TOKENS                = tostring(var.max_tokens)
      TRACING_ENDPOINT          = var.tracing_endpoint
      TRACING_PROJECT           = var.tracing_project
      TRACING_SPACE_ID          = var.tracing_space_id
      TRACING_API_KEY_PARAMETER = var.tracing_api_key_parameter
    }
  }

  depends_on = [aws_cloudwatch_log_group.chat, aws_iam_role_policy_attachment.chat_logs]
}

# ---------- HTTP API ----------

resource "aws_apigatewayv2_api" "this" {
  name          = "${var.name_prefix}-api"
  protocol_type = "HTTP"
  tags          = var.tags

  dynamic "cors_configuration" {
    for_each = length(var.cors_allow_origins) > 0 ? [1] : []
    content {
      allow_origins = var.cors_allow_origins
      allow_methods = ["POST", "OPTIONS"]
      allow_headers = ["authorization", "content-type"]
      max_age       = 3600
    }
  }
}

resource "aws_apigatewayv2_authorizer" "cognito" {
  api_id           = aws_apigatewayv2_api.this.id
  name             = "cognito"
  authorizer_type  = "JWT"
  identity_sources = ["$request.header.Authorization"]

  # Access tokens carry client_id instead of aud; API Gateway checks either against the audience.
  jwt_configuration {
    issuer   = var.jwt_issuer
    audience = var.jwt_audience
  }
}

resource "aws_apigatewayv2_integration" "chat" {
  api_id                 = aws_apigatewayv2_api.this.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.chat.invoke_arn
  payload_format_version = "2.0"
}

# The /api prefix matches the CloudFront path that forwards to this API.
resource "aws_apigatewayv2_route" "chat" {
  api_id             = aws_apigatewayv2_api.this.id
  route_key          = "POST /api/chat"
  target             = "integrations/${aws_apigatewayv2_integration.chat.id}"
  authorization_type = "JWT"
  authorizer_id      = aws_apigatewayv2_authorizer.cognito.id
}

resource "aws_cloudwatch_log_group" "access" {
  name              = "/aws/apigateway/${var.name_prefix}-api"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.this.id
  name        = "$default"
  auto_deploy = true
  tags        = var.tags

  default_route_settings {
    throttling_rate_limit  = var.throttling_rate_limit
    throttling_burst_limit = var.throttling_burst_limit
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.access.arn
    format = jsonencode({
      requestId = "$context.requestId"
      time      = "$context.requestTime"
      route     = "$context.routeKey"
      status    = "$context.status"
      latencyMs = "$context.responseLatency"
      user      = "$context.authorizer.claims.sub"
      authError = "$context.authorizer.error"
    })
  }
}

resource "aws_lambda_permission" "api" {
  statement_id  = "AllowApiGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chat.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.this.execution_arn}/*/*/api/chat"
}
