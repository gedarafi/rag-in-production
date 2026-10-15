variable "name_prefix" {
  description = "Prefix for every resource name, e.g. rag-aws."
  type        = string
}

variable "allowed_email_domains" {
  description = "Exact email domains that may register. Subdomains are not matched automatically."
  type        = list(string)
  default     = ["uu.se", "student.uu.se"]

  validation {
    condition     = length(var.allowed_email_domains) > 0 && alltrue([for d in var.allowed_email_domains : can(regex("^[a-z0-9.-]+\\.[a-z]{2,}$", d))])
    error_message = "Give at least one lowercase domain such as uu.se, without @."
  }
}

variable "ses_identity" {
  description = "Domain (e.g. uurag.se) or single email address that sends the one-time codes. Verified in SES by this module."
  type        = string

  validation {
    condition     = can(regex("^([^@\\s]+@)?[a-z0-9.-]+\\.[a-z]{2,}$", var.ses_identity))
    error_message = "ses_identity must be a lowercase domain or an email address."
  }
}

variable "create_ses_identity" {
  description = "Create and verify ses_identity in SES. Set to false when it already exists, or on LocalStack, which can't tag SESv2 identities."
  type        = bool
  default     = true
}

variable "route53_zone_id" {
  description = "Hosted zone for ses_identity when it is a domain. When set, the DKIM and DMARC records are created there."
  type        = string
  default     = null
}

variable "from_name" {
  description = "Display name on the code emails."
  type        = string
  default     = "RAG in Production"
}

variable "deletion_protection" {
  description = "Protect the user pool from deletion. Keep ACTIVE once real users exist."
  type        = string
  default     = "ACTIVE"

  validation {
    condition     = contains(["ACTIVE", "INACTIVE"], var.deletion_protection)
    error_message = "deletion_protection must be ACTIVE or INACTIVE."
  }
}

variable "log_retention_days" {
  description = "How long to keep the pre sign-up Lambda logs."
  type        = number
  default     = 14
}

variable "tags" {
  description = "Tags added to every resource."
  type        = map(string)
  default     = {}
}
