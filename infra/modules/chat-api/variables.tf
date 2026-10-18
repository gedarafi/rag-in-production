variable "name_prefix" {
  description = "Prefix for every resource name, e.g. rag-aws."
  type        = string
}

variable "lambda_source_dir" {
  description = "Folder zipped as the chat Lambda: handler.py from src/chat plus its dependencies (Terragrunt builds it in infra/.build/chat)."
  type        = string
}

variable "jwt_issuer" {
  description = "Token issuer URL (the auth module's issuer output)."
  type        = string

  validation {
    condition     = startswith(var.jwt_issuer, "https://")
    error_message = "jwt_issuer must be an https:// URL."
  }
}

variable "jwt_audience" {
  description = "Accepted app client IDs (the auth module's user_pool_client_id output)."
  type        = list(string)
}

variable "jobs_table" {
  description = "DynamoDB table of the chat jobs (recommender module)."
  type        = string
}

variable "jobs_table_arn" {
  description = "ARN of the chat jobs table."
  type        = string
}

variable "conversations_table" {
  description = "DynamoDB table of the conversations (recommender module)."
  type        = string
}

variable "queue_url" {
  description = "URL of the queue the questions go to (recommender module)."
  type        = string
}

variable "queue_arn" {
  description = "ARN of the queue the questions go to."
  type        = string
}

variable "tracing_endpoint" {
  description = "OTLP/HTTP endpoint the Lambda sends its traces to, e.g. a Phoenix at http://phoenix:6006/v1/traces or Arize AX at https://otlp.arize.com/v1/traces. Empty turns tracing off."
  type        = string
  default     = ""
}

variable "tracing_project" {
  description = "Project the traces are filed under in Phoenix or Arize."
  type        = string
  default     = "rag-in-production"
}

variable "tracing_space_id" {
  description = "Arize space ID, sent as the arize-space-id header. Empty for Phoenix."
  type        = string
  default     = ""
}

variable "tracing_api_key_parameter" {
  description = "Name of the SSM SecureString holding the Arize API key, e.g. /rag-aws/arize-api-key. The Lambda may read only this parameter. Empty for Phoenix."
  type        = string
  default     = ""

  validation {
    condition     = var.tracing_api_key_parameter == "" || startswith(var.tracing_api_key_parameter, "/")
    error_message = "tracing_api_key_parameter must be a full SSM name starting with /."
  }
}

variable "throttling_rate_limit" {
  description = "Steady-state requests per second for the whole API. A cheap guard against runaway costs."
  type        = number
  default     = 5
}

variable "throttling_burst_limit" {
  description = "Maximum burst of concurrent requests for the whole API."
  type        = number
  default     = 10
}

variable "cors_allow_origins" {
  description = "Origins allowed to call the API directly, e.g. http://localhost:8000 for local frontend work. The deployed page calls the API through CloudFront on the same origin and needs none."
  type        = list(string)
  default     = []
}

variable "log_retention_days" {
  description = "How long to keep the Lambda and API access logs."
  type        = number
  default     = 14
}

variable "tags" {
  description = "Tags added to every resource."
  type        = map(string)
  default     = {}
}
