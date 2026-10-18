# The anime recommender's backend: the api Lambda (chat-api module) queues each question in SQS; the
# worker Lambda here (a container image from the container-registry unit) answers it from the artifact
# bundle in S3 with a Bedrock model, and stores the answer and the conversation in DynamoDB.

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  worker_name = "${var.name_prefix}-recommender"

  # A cross-region inference profile (us.amazon.nova-micro-v1:0) routes to the foundation model
  # (amazon.nova-micro-v1:0) in any region of its geography; IAM must allow both.
  base_model_id = try(regex("^(?:us|eu|apac|jp|au|ca|us-gov|global)\\.(.+)$", var.model_id)[0], var.model_id)

  # Converse authorizes the model. bedrock-mantle authorizes the project the request runs in: the
  # account's default project, since the worker names none.
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
    # The deterministic stand-in (CI) and a local Ollama (LocalStack) need no Bedrock permission.
    stub = {
      Effect   = "Deny"
      Action   = "bedrock:*"
      Resource = "*"
    }
    ollama = {
      Effect   = "Deny"
      Action   = "bedrock:*"
      Resource = "*"
    }
  }[var.model_api]

  parameter_arn = "arn:aws:ssm:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:parameter"
}

# ---------- Artifact bundles ----------

module "artifacts" {
  source        = "../s3-bucket"
  name          = var.artifact_bucket_name
  force_destroy = var.force_destroy
  tags          = var.tags
}

# Which bundle version the worker serves. `recommender publish` and `recommender activate` change the
# value, so OpenTofu only creates the parameter.
resource "aws_ssm_parameter" "artifact_version" {
  name        = "/${var.name_prefix}/recommender/artifact-version"
  description = "Version of the recommender artifact bundle the worker serves (recommender/<version>/ in the artifacts bucket)."
  type        = "String"
  value       = "none"
  tags        = var.tags

  lifecycle {
    ignore_changes = [value]
  }
}

# ---------- Jobs and conversations ----------

resource "aws_dynamodb_table" "jobs" {
  name         = "${var.name_prefix}-chat-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"
  tags         = var.tags

  attribute {
    name = "job_id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  server_side_encryption {
    enabled = true
  }
}

resource "aws_dynamodb_table" "conversations" {
  name         = "${var.name_prefix}-chat-conversations"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "turn"
  tags         = var.tags

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "turn"
    type = "N"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = var.point_in_time_recovery
  }

  server_side_encryption {
    enabled = true
  }
}

# ---------- Queue ----------

resource "aws_sqs_queue" "dead_letters" {
  name                      = "${local.worker_name}-dlq"
  message_retention_seconds = 14 * 24 * 3600
  sqs_managed_sse_enabled   = true
  tags                      = var.tags
}

resource "aws_sqs_queue" "jobs" {
  name = "${local.worker_name}-jobs"
  # AWS recommends at least six times the function timeout, so a slow attempt is never delivered twice.
  visibility_timeout_seconds = 6 * var.timeout_seconds
  message_retention_seconds  = 3600 # the page stops waiting long before
  sqs_managed_sse_enabled    = true
  tags                       = var.tags

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letters.arn
    maxReceiveCount     = var.max_receives
  })
}

# ---------- Worker Lambda ----------

resource "aws_iam_role" "worker" {
  name = local.worker_name
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

resource "aws_iam_role_policy_attachment" "worker_logs" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "worker" {
  name = "recommender"
  role = aws_iam_role.worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      local.bedrock_statement,
      {
        Sid      = "ReadBundles"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "${module.artifacts.bucket_arn}/recommender/*"
      },
      {
        Sid      = "ChatData"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query"]
        Resource = [aws_dynamodb_table.jobs.arn, aws_dynamodb_table.conversations.arn]
      },
      {
        Sid      = "ConsumeJobs"
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"]
        Resource = aws_sqs_queue.jobs.arn
      },
      {
        Sid      = "ReadActiveVersion"
        Effect   = "Allow"
        Action   = "ssm:GetParameter"
        Resource = aws_ssm_parameter.artifact_version.arn
      },
      ], var.tracing_api_key_parameter == "" ? [] : [{
        # The tracing API key (Arize AX) stays in SSM, out of the environment and the state.
        Sid      = "ReadTracingKey"
        Effect   = "Allow"
        Action   = "ssm:GetParameter"
        Resource = "${local.parameter_arn}/${trimprefix(var.tracing_api_key_parameter, "/")}"
    }])
  })
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/lambda/${local.worker_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_lambda_function" "worker" {
  function_name = local.worker_name
  role          = aws_iam_role.worker.arn
  package_type  = "Image"
  # Pinning the digest the build pushed makes every new build a new deployment. Without one (a
  # destroy, or a plan before the first push) the tag stands in, so no image needs to exist.
  image_uri     = var.image_digest == "" ? "${var.repository_url}:${var.image_tag}" : "${var.repository_url}@${var.image_digest}"
  architectures = ["x86_64"]
  timeout       = var.timeout_seconds
  memory_size   = var.memory_mb
  tags          = var.tags

  # The bundle is downloaded into /tmp at cold start and kept for the container's lifetime.
  ephemeral_storage {
    size = var.ephemeral_storage_mb
  }

  environment {
    variables = {
      MODEL_ID                   = var.model_id
      MODEL_API                  = var.model_api
      MODEL_TIMEOUT_SECONDS      = tostring(var.model_timeout_seconds)
      OLLAMA_URL                 = var.ollama_url
      JOBS_TABLE                 = aws_dynamodb_table.jobs.name
      CONVERSATIONS_TABLE        = aws_dynamodb_table.conversations.name
      ARTIFACT_BUCKET            = module.artifacts.bucket_name
      ARTIFACT_VERSION_PARAMETER = aws_ssm_parameter.artifact_version.name
      MAX_RECEIVES               = tostring(var.max_receives)
      TRACING_ENDPOINT           = var.tracing_endpoint
      TRACING_PROJECT            = var.tracing_project
      TRACING_SPACE_ID           = var.tracing_space_id
      TRACING_API_KEY_PARAMETER  = var.tracing_api_key_parameter
    }
  }

  depends_on = [aws_cloudwatch_log_group.worker, aws_iam_role_policy_attachment.worker_logs]
}

resource "aws_lambda_event_source_mapping" "jobs" {
  event_source_arn        = aws_sqs_queue.jobs.arn
  function_name           = aws_lambda_function.worker.arn
  batch_size              = 1
  function_response_types = ["ReportBatchItemFailures"]

  # Caps parallel answers (and Bedrock spend) without reserving account concurrency.
  scaling_config {
    maximum_concurrency = var.max_concurrency
  }
}

# ---------- Alarms ----------

resource "aws_sns_topic" "alarms" {
  name = "${local.worker_name}-alarms"
  tags = var.tags
}

resource "aws_sns_topic_subscription" "alarm_emails" {
  for_each  = toset(var.alarm_emails)
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = each.value
}

resource "aws_cloudwatch_metric_alarm" "dead_letters" {
  alarm_name          = "${local.worker_name}-dead-letters"
  alarm_description   = "A question failed every attempt and is in ${aws_sqs_queue.dead_letters.name}; see the worker's log and trace."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.dead_letters.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "worker_errors" {
  alarm_name          = "${local.worker_name}-errors"
  alarm_description   = "The recommender worker failed; see /aws/lambda/${local.worker_name}."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.worker.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "worker_slow" {
  alarm_name          = "${local.worker_name}-slow"
  alarm_description   = "Answers take more than 80 % of the worker's timeout at p95."
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  dimensions          = { FunctionName = aws_lambda_function.worker.function_name }
  extended_statistic  = "p95"
  period              = 900
  evaluation_periods  = 1
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.timeout_seconds * 1000 * 0.8
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags
}
