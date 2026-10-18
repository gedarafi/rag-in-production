output "queue_url" {
  description = "URL of the queue the api Lambda sends questions to."
  value       = aws_sqs_queue.jobs.url
}

output "queue_arn" {
  description = "ARN of the queue the api Lambda sends questions to."
  value       = aws_sqs_queue.jobs.arn
}

output "dead_letter_queue_url" {
  description = "Questions that failed every attempt."
  value       = aws_sqs_queue.dead_letters.url
}

output "jobs_table" {
  description = "DynamoDB table of the chat jobs."
  value       = aws_dynamodb_table.jobs.name
}

output "jobs_table_arn" {
  description = "ARN of the chat jobs table."
  value       = aws_dynamodb_table.jobs.arn
}

output "conversations_table" {
  description = "DynamoDB table of the conversations."
  value       = aws_dynamodb_table.conversations.name
}

output "conversations_table_arn" {
  description = "ARN of the conversations table."
  value       = aws_dynamodb_table.conversations.arn
}

output "artifact_bucket" {
  description = "Bucket to publish bundles to (`recommender publish`)."
  value       = module.artifacts.bucket_name
}

output "artifact_version_parameter" {
  description = "SSM parameter naming the bundle version the worker serves."
  value       = aws_ssm_parameter.artifact_version.name
}

output "worker_function_name" {
  description = "Name of the worker Lambda."
  value       = aws_lambda_function.worker.function_name
}

output "model_id" {
  description = "Bedrock model or inference profile the worker calls."
  value       = var.model_id
}

output "model_api" {
  description = "How the worker calls the model: converse (bedrock-runtime) or openai (bedrock-mantle)."
  value       = var.model_api
}

output "alarm_topic_arn" {
  description = "SNS topic of the worker's alarms."
  value       = aws_sns_topic.alarms.arn
}

output "vector_db_cluster_arn" {
  description = "Aurora cluster with the passage index (`recommender publish` loads it), or empty."
  value       = var.vector_db == null ? "" : var.vector_db.cluster_arn
}

output "vector_db_secret_arn" {
  description = "Secret the Data API signs in to the passage index with, or empty."
  value       = var.vector_db == null ? "" : var.vector_db.secret_arn
}

output "vector_db_name" {
  description = "Database of the passage index, or empty."
  value       = var.vector_db == null ? "" : var.vector_db.database
}
