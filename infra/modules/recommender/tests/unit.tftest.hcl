# Plan-only tests with mocked providers. They need no LocalStack and no credentials.

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
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/rag-unit-recommender"
    }
  }
  mock_resource "aws_sqs_queue" {
    defaults = {
      arn = "arn:aws:sqs:us-east-2:123456789012:rag-unit-recommender-jobs"
      url = "https://sqs.us-east-2.amazonaws.com/123456789012/rag-unit-recommender-jobs"
    }
  }
  mock_resource "aws_dynamodb_table" {
    defaults = {
      arn = "arn:aws:dynamodb:us-east-2:123456789012:table/rag-unit-chat"
    }
  }
  mock_resource "aws_s3_bucket" {
    defaults = {
      arn = "arn:aws:s3:::rag-unit-artifacts"
    }
  }
  mock_resource "aws_ssm_parameter" {
    defaults = {
      arn = "arn:aws:ssm:us-east-2:123456789012:parameter/rag-unit/recommender/artifact-version"
    }
  }
  mock_resource "aws_lambda_function" {
    defaults = {
      arn = "arn:aws:lambda:us-east-2:123456789012:function:rag-unit-recommender"
    }
  }
  mock_resource "aws_sns_topic" {
    defaults = {
      arn = "arn:aws:sns:us-east-2:123456789012:rag-unit-recommender-alarms"
    }
  }
}

variables {
  name_prefix          = "rag-unit"
  repository_name      = "rag-unit-recommender"
  repository_url       = "123456789012.dkr.ecr.us-east-2.amazonaws.com/rag-unit-recommender"
  artifact_bucket_name = "rag-unit-artifacts"
  model_id             = "qwen.qwen3-32b"
  model_api            = "openai"
  image_digest         = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}

run "runs_the_pinned_image" {
  command = plan

  assert {
    condition     = aws_lambda_function.worker.image_uri == "123456789012.dkr.ecr.us-east-2.amazonaws.com/rag-unit-recommender@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    error_message = "The worker must run the image by digest, so each build is a new deployment."
  }

  assert {
    condition     = aws_lambda_function.worker.ephemeral_storage[0].size == 6144
    error_message = "/tmp must fit the extracted bundle."
  }
}

run "falls_back_to_the_tag_without_a_digest" {
  command = plan

  variables {
    image_digest = ""
  }

  assert {
    condition     = aws_lambda_function.worker.image_uri == "123456789012.dkr.ecr.us-east-2.amazonaws.com/rag-unit-recommender:current"
    error_message = "Without a pushed digest (destroy, first plan) the worker names the tag, so no image lookup is needed."
  }
}

run "queues_with_a_dead_letter_queue" {
  command = plan

  assert {
    condition     = jsondecode(aws_sqs_queue.jobs.redrive_policy).maxReceiveCount == 2
    error_message = "A question gets two attempts before the dead-letter queue."
  }

  assert {
    condition     = aws_sqs_queue.jobs.visibility_timeout_seconds == 6 * 120
    error_message = "The visibility timeout must be six times the worker timeout."
  }

  assert {
    condition     = aws_lambda_event_source_mapping.jobs.batch_size == 1 && aws_lambda_event_source_mapping.jobs.scaling_config[0].maximum_concurrency == 2
    error_message = "One question per invocation, at most two at once."
  }

  assert {
    condition     = contains(aws_lambda_event_source_mapping.jobs.function_response_types, "ReportBatchItemFailures")
    error_message = "Failed messages are reported one by one."
  }
}

run "keeps_messages_until_their_last_attempt" {
  command = plan

  # The local stack's worker timeout: its visibility timeout is 90 minutes.
  variables {
    timeout_seconds = 900
  }

  assert {
    condition     = aws_sqs_queue.jobs.message_retention_seconds >= aws_sqs_queue.jobs.visibility_timeout_seconds * 2
    error_message = "A failed attempt must still be in the queue to be retried and then dead-lettered."
  }
}

run "least_privilege" {
  command = plan

  assert {
    condition = jsondecode(aws_iam_role_policy.worker.policy).Statement[0] == {
      Effect   = "Allow"
      Action   = "bedrock-mantle:CreateInference"
      Resource = ["arn:aws:bedrock-mantle:us-east-2:123456789012:project/default"]
    }
    error_message = "With model_api openai the worker may only call bedrock-mantle."
  }

  assert {
    condition     = one([for s in jsondecode(aws_iam_role_policy.worker.policy).Statement : s.Resource if try(s.Sid, "") == "ReadBundles"]) == "arn:aws:s3:::rag-unit-artifacts/recommender/*"
    error_message = "The worker may only read the bundles."
  }

  assert {
    condition     = length([for s in jsondecode(aws_iam_role_policy.worker.policy).Statement : s if try(s.Sid, "") == "ReadTracingKey"]) == 0
    error_message = "Without an Arize key parameter the worker reads no secret."
  }
}

run "reads_the_tracing_key_only_when_set" {
  command = plan

  variables {
    tracing_api_key_parameter = "/rag-aws/arize-api-key"
  }

  assert {
    condition     = one([for s in jsondecode(aws_iam_role_policy.worker.policy).Statement : s.Resource if try(s.Sid, "") == "ReadTracingKey"]) == "arn:aws:ssm:us-east-2:123456789012:parameter/rag-aws/arize-api-key"
    error_message = "The worker may read exactly the Arize key parameter."
  }
}

run "conversations_expire" {
  command = plan

  assert {
    condition     = aws_dynamodb_table.conversations.ttl[0].attribute_name == "expires_at" && aws_dynamodb_table.jobs.ttl[0].attribute_name == "expires_at"
    error_message = "Jobs and conversations must expire (TTL expires_at)."
  }

  assert {
    condition     = aws_dynamodb_table.conversations.billing_mode == "PAY_PER_REQUEST"
    error_message = "Tables are billed per request; there is no steady traffic."
  }
}

run "alarms_mail_the_given_addresses" {
  command = plan

  variables {
    alarm_emails = ["ops@example.com"]
  }

  assert {
    condition     = aws_sns_topic_subscription.alarm_emails["ops@example.com"].protocol == "email"
    error_message = "Alarm emails must be subscribed to the topic."
  }

  assert {
    condition     = aws_cloudwatch_metric_alarm.worker_slow.threshold == 96000
    error_message = "The slow alarm fires at 80 % of the timeout."
  }
}

run "rejects_tiny_storage" {
  command = plan

  variables {
    ephemeral_storage_mb = 100
  }

  expect_failures = [var.ephemeral_storage_mb]
}

run "may_invoke_only_the_configured_model" {
  command = plan

  variables {
    model_id  = "us.amazon.nova-micro-v1:0"
    model_api = "converse"
  }

  assert {
    condition = jsondecode(aws_iam_role_policy.worker.policy).Statement[0].Resource == [
      "arn:aws:bedrock:*::foundation-model/amazon.nova-micro-v1:0",
      "arn:aws:bedrock:us-east-2:123456789012:inference-profile/us.amazon.nova-micro-v1:0",
    ]
    error_message = "The worker may invoke the inference profile and its foundation model, nothing else."
  }

  assert {
    condition     = aws_lambda_function.worker.environment[0].variables.MODEL_API == "converse"
    error_message = "The worker must call the API it is configured for."
  }
}

run "allows_a_plain_foundation_model" {
  command = plan

  variables {
    model_id  = "amazon.nova-lite-v1:0"
    model_api = "converse"
  }

  assert {
    condition     = contains(jsondecode(aws_iam_role_policy.worker.policy).Statement[0].Resource, "arn:aws:bedrock:*::foundation-model/amazon.nova-lite-v1:0")
    error_message = "A model ID without a geography prefix is the foundation model itself."
  }
}

run "rejects_an_unknown_model_api" {
  command = plan

  variables {
    model_api = "invoke"
  }

  expect_failures = [var.model_api]
}

run "the_stub_may_call_no_model" {
  command = plan

  variables {
    model_id  = "stub"
    model_api = "stub"
  }

  assert {
    condition     = jsondecode(aws_iam_role_policy.worker.policy).Statement[0].Effect == "Deny"
    error_message = "With the stand-in model the worker may not call Bedrock at all."
  }
}

run "a_local_ollama_needs_no_bedrock" {
  command = plan

  variables {
    model_id   = "llama3.2:3b"
    model_api  = "ollama"
    ollama_url = "http://ollama:11434"
  }

  assert {
    condition     = aws_lambda_function.worker.environment[0].variables.OLLAMA_URL == "http://ollama:11434"
    error_message = "The worker must know where the local Ollama is."
  }

  assert {
    condition     = jsondecode(aws_iam_role_policy.worker.policy).Statement[0].Effect == "Deny"
    error_message = "With a local Ollama the worker may not call Bedrock at all."
  }
}

run "queries_the_passage_index_only_when_given" {
  command = plan

  assert {
    condition     = aws_lambda_function.worker.environment[0].variables.VECTOR_DB_CLUSTER_ARN == "" && !strcontains(aws_iam_role_policy.worker.policy, "rds-data")
    error_message = "Without a vector database the worker gets neither its address nor access."
  }
}

run "may_query_the_given_passage_index" {
  command = plan

  variables {
    vector_db = {
      cluster_arn = "arn:aws:rds:us-east-2:123456789012:cluster:rag-unit-vectors"
      secret_arn  = "arn:aws:secretsmanager:us-east-2:123456789012:secret:rds-rag-unit-vectors"
      database    = "vectors"
    }
  }

  assert {
    condition     = aws_lambda_function.worker.environment[0].variables.VECTOR_DB_CLUSTER_ARN == "arn:aws:rds:us-east-2:123456789012:cluster:rag-unit-vectors" && aws_lambda_function.worker.environment[0].variables.VECTOR_DB_NAME == "vectors"
    error_message = "The worker needs the cluster, the secret and the database of the passage index."
  }

  assert {
    condition = anytrue([for statement in jsondecode(aws_iam_role_policy.worker.policy).Statement :
    statement.Action == "rds-data:ExecuteStatement" && statement.Resource == "arn:aws:rds:us-east-2:123456789012:cluster:rag-unit-vectors"])
    error_message = "The worker may query only the given cluster."
  }

  assert {
    condition = anytrue([for statement in jsondecode(aws_iam_role_policy.worker.policy).Statement :
    statement.Action == "secretsmanager:GetSecretValue" && statement.Resource == "arn:aws:secretsmanager:us-east-2:123456789012:secret:rds-rag-unit-vectors"])
    error_message = "The worker may read only the cluster's master secret."
  }
}
