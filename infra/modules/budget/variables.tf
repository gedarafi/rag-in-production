variable "name" {
  description = "Budget name, unique in the account."
  type        = string

  validation {
    condition     = length(trimspace(var.name)) > 0
    error_message = "name must not be empty."
  }
}

variable "limit_usd" {
  description = "Monthly cost limit in USD for the whole account."
  type        = number

  validation {
    condition     = var.limit_usd > 0
    error_message = "limit_usd must be greater than 0."
  }
}

variable "emails" {
  description = "Addresses emailed when spend passes 80 % of the limit or is forecast to pass it."
  type        = list(string)

  validation {
    condition     = length(var.emails) > 0 && alltrue([for e in var.emails : can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", e))])
    error_message = "emails must list at least one email address (set RAG_BUDGET_EMAILS for real AWS)."
  }
}

variable "tags" {
  description = "Tags added to the budget."
  type        = map(string)
  default     = {}
}
