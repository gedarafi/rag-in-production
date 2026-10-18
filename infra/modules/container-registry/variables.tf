variable "name" {
  description = "Repository name, e.g. rag-local-recommender."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]+(?:[._-][a-z0-9]+)*$", var.name))
    error_message = "The name must be lowercase letters and digits, separated by single dots, hyphens or underscores."
  }
}

variable "keep_images" {
  description = "How many tagged builds (build-<hash>) to keep."
  type        = number
  default     = 5
}

variable "force_delete" {
  description = "Delete the images with the repository. Only enable outside production."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags added to the repository."
  type        = map(string)
  default     = {}
}
