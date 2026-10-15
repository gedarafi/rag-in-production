variable "name" {
  description = "Bucket name. Must follow the S3 bucket naming rules."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.name)) && !strcontains(var.name, "..")
    error_message = "The name must be 3-63 characters of lowercase letters, digits, dots and hyphens, and start and end with a letter or digit."
  }
}

variable "force_destroy" {
  description = "Delete all objects when the bucket is destroyed. Only enable outside production."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags added to the bucket."
  type        = map(string)
  default     = {}
}
