variable "name" {
  description = "Name of the cluster and its network, e.g. rag-local-vectors."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,50}[a-z0-9]$", var.name))
    error_message = "The name must start with a letter and contain only lowercase letters, digits and hyphens."
  }
}

variable "database" {
  description = "Name of the database the Data API connects to."
  type        = string
  default     = "vectors"
}

variable "engine_version" {
  description = "Aurora PostgreSQL version; scaling to 0 ACU needs 16.3 or later, halfvec needs pgvector 0.7 or later."
  type        = string
  default     = "16.15"
}

variable "max_capacity" {
  description = "Most Aurora capacity units (2 GiB of memory each) the cluster may scale to."
  type        = number
  default     = 2

  validation {
    condition     = var.max_capacity >= 1 && var.max_capacity <= 16
    error_message = "max_capacity must be between 1 and 16 ACU."
  }
}

variable "seconds_until_auto_pause" {
  description = "Idle seconds before the cluster pauses at 0 ACU (300 to 86400)."
  type        = number
  default     = 300

  validation {
    condition     = var.seconds_until_auto_pause >= 300 && var.seconds_until_auto_pause <= 86400
    error_message = "seconds_until_auto_pause must be between 300 and 86400."
  }
}

variable "cidr_block" {
  description = "Address range of the cluster's own VPC; split into its two subnets."
  type        = string
  default     = "10.42.0.0/24"
}

variable "deletion_protection" {
  description = "Refuse to delete the cluster."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags added to every resource."
  type        = map(string)
  default     = {}
}
