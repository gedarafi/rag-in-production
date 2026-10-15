# Creates the bucket on LocalStack, checks it, and destroys it. Start LocalStack first (`just up`).

provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  s3_use_path_style           = true
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true

  endpoints {
    s3  = "http://localhost:4566"
    sts = "http://localhost:4566"
  }
}

variables {
  name          = "rag-localstack-test"
  force_destroy = true
}

run "creates_the_bucket" {
  command = apply

  assert {
    condition     = output.bucket_name == "rag-localstack-test"
    error_message = "The created bucket must use the given name."
  }

  assert {
    condition     = output.bucket_arn == "arn:aws:s3:::rag-localstack-test"
    error_message = "The bucket ARN must match the name."
  }

  assert {
    condition     = aws_s3_bucket_versioning.this.versioning_configuration[0].status == "Enabled"
    error_message = "Versioning must be enabled on the created bucket."
  }
}
