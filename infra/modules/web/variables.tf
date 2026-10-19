variable "name_prefix" {
  description = "Prefix for every resource name, e.g. rag-aws."
  type        = string
}

variable "site_dir" {
  description = "Folder with the static page (index.html, app.js, style.css). config.js is generated here."
  type        = string
}

variable "region" {
  description = "Region of the Cognito user pool, used by the page to call Cognito."
  type        = string
}

variable "user_pool_client_id" {
  description = "Public Cognito app client ID (the auth module's user_pool_client_id output)."
  type        = string
}

variable "allowed_email_domains" {
  description = "Domains shown and pre-checked on the sign-in page. The real check is in Cognito."
  type        = list(string)
}

variable "api_domain" {
  description = "Host name of the HTTP API (the chat-api module's api_domain output). Served under /api/*."
  type        = string
}

variable "api_origin_port" {
  description = "Port of the HTTP API origin. 443 on AWS; LocalStack serves it on 4566."
  type        = number
  default     = 443
}

variable "api_origin_protocol_policy" {
  description = "How CloudFront connects to the API: https-only on AWS, http-only on LocalStack."
  type        = string
  default     = "https-only"

  validation {
    condition     = contains(["https-only", "http-only"], var.api_origin_protocol_policy)
    error_message = "api_origin_protocol_policy must be https-only or http-only."
  }
}

variable "recommender_origin_domain" {
  description = "Public DNS name of the recommender instance (the recommender-host module's origin_domain). Served under /recommend/*; empty leaves the route out."
  type        = string
  default     = ""
}

variable "recommender_origin_port" {
  description = "Port of the recommender service on its origin (plain HTTP)."
  type        = number
  default     = 8080
}

variable "price_class" {
  description = "CloudFront price class. PriceClass_100 serves from Europe and North America, the cheapest."
  type        = string
  default     = "PriceClass_100"
}

variable "force_destroy" {
  description = "Delete the site files when the bucket is destroyed."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags added to every resource."
  type        = map(string)
  default     = {}
}
