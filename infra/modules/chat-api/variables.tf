variable "name_prefix" {
  description = "Prefix for every resource name, e.g. rag-aws."
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
