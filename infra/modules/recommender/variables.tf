variable "name_prefix" {
  description = "Prefix of every resource name, e.g. rag-local or rag-aws."
  type        = string
}

variable "repository_name" {
  description = "ECR repository of the worker image (container-registry unit)."
  type        = string
}

variable "repository_url" {
  description = "ECR repository URL of the worker image (container-registry unit)."
  type        = string
}

variable "image_tag" {
  description = "Tag the image build moves to each new image; used only when image_digest is empty."
  type        = string
  default     = "current"
}

variable "image_digest" {
  description = "Digest of the image to run (sha256:...). The build hook writes it to image.auto.tfvars before plan and apply."
  type        = string
  default     = ""

  validation {
    condition     = var.image_digest == "" || can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "image_digest must be empty or sha256:<64 hex digits>."
  }
}

variable "artifact_bucket_name" {
  description = "Bucket of the artifact bundles (recommender/<version>/)."
  type        = string
}

variable "model_id" {
  description = "Bedrock model or inference profile the worker answers with, e.g. qwen.qwen3-32b or, on LocalStack, ollama.llama3.2:3b."
  type        = string
}

variable "model_api" {
  description = "How the worker calls the model: converse (Converse on bedrock-runtime), openai (Chat Completions on bedrock-mantle), ollama (a local Ollama with schema-constrained output, LocalStack only) or stub (a deterministic stand-in for CI)."
  type        = string
  default     = "converse"

  validation {
    condition     = contains(["converse", "openai", "ollama", "stub"], var.model_api)
    error_message = "model_api must be converse, openai, ollama or stub."
  }
}

variable "ollama_url" {
  description = "Ollama the worker calls with model_api ollama, e.g. http://ollama:11434 on the Compose network."
  type        = string
  default     = ""
}

variable "vector_db" {
  description = "The Aurora cluster with the passage index (modules/vector-db outputs); null leaves the worker without semantic passage search."
  type = object({
    cluster_arn = string
    secret_arn  = string
    database    = string
  })
  default = null
}

variable "model_timeout_seconds" {
  description = "How long one model call may take before it counts as failed."
  type        = number
  default     = 60
}

variable "timeout_seconds" {
  description = "Worker timeout: an answer takes several model calls, and the first one after a cold start also downloads the bundle."
  type        = number
  default     = 120

  validation {
    condition     = var.timeout_seconds >= 30 && var.timeout_seconds <= 900
    error_message = "timeout_seconds must be between 30 and 900."
  }
}

variable "memory_mb" {
  description = "Worker memory; CPU (for the embedding model) scales with it."
  type        = number
  default     = 2048
}

variable "ephemeral_storage_mb" {
  description = "Worker /tmp, which holds the extracted bundle (about 2 GB for the full AniList catalog)."
  type        = number
  default     = 6144

  validation {
    condition     = var.ephemeral_storage_mb >= 512 && var.ephemeral_storage_mb <= 10240
    error_message = "ephemeral_storage_mb must be between 512 and 10240."
  }
}

variable "max_concurrency" {
  description = "Most questions answered at once (at least 2, an SQS event source limit); caps Bedrock spend."
  type        = number
  default     = 2

  validation {
    condition     = var.max_concurrency >= 2
    error_message = "max_concurrency must be at least 2."
  }
}

variable "max_receives" {
  description = "Attempts per question before it goes to the dead-letter queue."
  type        = number
  default     = 2
}

variable "point_in_time_recovery" {
  description = "Continuous backups of the conversations table (real AWS)."
  type        = bool
  default     = false
}

variable "alarm_emails" {
  description = "Addresses that get the alarms (each confirms an SNS subscription email). Empty sends none."
  type        = list(string)
  default     = []
}

variable "tracing_endpoint" {
  description = "OTLP/HTTP endpoint for the worker's traces, e.g. http://phoenix:6006/v1/traces or https://otlp.arize.com/v1/traces. Empty turns tracing off."
  type        = string
  default     = ""
}

variable "tracing_project" {
  description = "Project the traces are filed under in Phoenix or Arize."
  type        = string
  default     = "rag-in-production"
}

variable "tracing_space_id" {
  description = "Arize space ID; empty for Phoenix."
  type        = string
  default     = ""
}

variable "tracing_api_key_parameter" {
  description = "SSM SecureString with the Arize API key, e.g. /rag-aws/arize-api-key; empty for Phoenix."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "How long the worker's logs are kept."
  type        = number
  default     = 14
}

variable "force_destroy" {
  description = "Delete the bundles with the bucket. Only enable outside production."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags added to every resource."
  type        = map(string)
  default     = {}
}
