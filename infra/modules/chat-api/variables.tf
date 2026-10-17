variable "name_prefix" {
  description = "Prefix for every resource name, e.g. rag-aws."
  type        = string
}

variable "lambda_source_dir" {
  description = "Folder with the chat Lambda code: src/chat in the repository (Terragrunt passes it in)."
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

variable "model_id" {
  description = "Bedrock model that answers: for converse a model or inference profile, e.g. us.amazon.nova-micro-v1:0; for openai a bedrock-mantle model, e.g. google.gemma-3-4b-it. On LocalStack, ollama.<model> runs that Ollama model on CPU."
  type        = string

  validation {
    condition     = length(trimspace(var.model_id)) > 0
    error_message = "model_id must not be empty."
  }
}

variable "model_api" {
  description = "How the Lambda calls the model: converse (Converse on bedrock-runtime, the only one LocalStack emulates) or openai (OpenAI-compatible Chat Completions on bedrock-mantle, which has its own quotas)."
  type        = string
  default     = "converse"

  validation {
    condition     = contains(["converse", "openai"], var.model_api)
    error_message = "model_api must be converse or openai."
  }
}

variable "max_tokens" {
  description = "Longest answer the model may write, in tokens."
  type        = number
  default     = 512

  validation {
    condition     = var.max_tokens >= 1 && var.max_tokens <= 4096
    error_message = "max_tokens must be between 1 and 4096."
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
