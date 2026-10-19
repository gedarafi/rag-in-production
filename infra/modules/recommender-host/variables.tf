variable "name_prefix" {
  description = "Prefix for resource names, e.g. rag-aws."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type. The models run on Bedrock, so memory sizes it: t4g.medium (4 GB) fits about 30 chatting users."
  type        = string
  default     = "t4g.medium"
}

variable "create_instance" {
  description = "Whether to run the instance. The bucket, role and security group exist either way, so the library can be published first."
  type        = bool
  default     = true
}

variable "ami_id" {
  description = "AMI to boot. Empty looks up the latest Amazon Linux 2023 for the instance's architecture."
  type        = string
  default     = ""
}

variable "root_volume_gb" {
  description = "Root disk size: the OS, Python, PyTorch, and two library versions (about 2 GB each)."
  type        = number
  default     = 20
}

variable "port" {
  description = "Port the service listens on; CloudFront connects to it over HTTP."
  type        = number
  default     = 8080
}

variable "workers" {
  description = "Questions the service answers at once; more wait up to 30 s, then get a busy reply."
  type        = number
  default     = 8
}

variable "cognito_issuer" {
  description = "Issuer of the user pool's tokens (the auth module's issuer output)."
  type        = string
}

variable "cognito_client_id" {
  description = "App client whose access tokens are accepted (the auth module's user_pool_client_id)."
  type        = string
}

variable "bedrock_model" {
  description = "Model for every LLM step, called through bedrock-mantle Chat Completions. It must follow JSON schemas."
  type        = string
  default     = "qwen.qwen3-32b"
}

variable "ingress_cidr_blocks" {
  description = "Who may reach the port. Empty allows only CloudFront (its origin-facing managed prefix list)."
  type        = list(string)
  default     = []
}

variable "tracing_endpoint" {
  description = "OTLP/HTTP endpoint the service sends its traces to, e.g. Arize AX at https://otlp.arize.com/v1/traces. Empty turns tracing off."
  type        = string
  default     = ""
}

variable "tracing_project" {
  description = "Project the traces are filed under in Arize (or Phoenix)."
  type        = string
  default     = "rag-in-production"
}

variable "tracing_space_id" {
  description = "Arize space ID, sent as the arize-space-id header. Empty for Phoenix."
  type        = string
  default     = ""
}

variable "tracing_api_key_parameter" {
  description = "Name of the SSM SecureString holding the Arize API key, e.g. /rag-aws/arize-api-key. The instance may read only this parameter. Empty for Phoenix."
  type        = string
  default     = ""

  validation {
    condition     = var.tracing_api_key_parameter == "" || startswith(var.tracing_api_key_parameter, "/")
    error_message = "tracing_api_key_parameter must be a full SSM name starting with /."
  }
}

variable "wheel" {
  description = "File name of the package wheel under releases/ in the data bucket (`just recommender-wheel`)."
  type        = string
  default     = "rag_in_production-0.1.0-py3-none-any.whl"
}

variable "force_destroy" {
  description = "Allow destroying the data bucket while it still holds published libraries."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags added to every resource."
  type        = map(string)
  default     = {}
}
