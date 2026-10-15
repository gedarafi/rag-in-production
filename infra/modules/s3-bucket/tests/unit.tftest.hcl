# Plan-only tests with a mocked AWS provider. They need no LocalStack and no credentials.

mock_provider "aws" {}

variables {
  name = "rag-unit-test"
  tags = { environment = "test" }
}

run "configures_a_private_versioned_encrypted_bucket" {
  command = plan

  assert {
    condition     = aws_s3_bucket.this.bucket == "rag-unit-test"
    error_message = "The bucket must use the given name."
  }

  assert {
    condition     = aws_s3_bucket.this.force_destroy == false
    error_message = "force_destroy must be off by default."
  }

  assert {
    condition     = aws_s3_bucket.this.tags["environment"] == "test"
    error_message = "The bucket must carry the given tags."
  }

  assert {
    condition     = aws_s3_bucket_versioning.this.versioning_configuration[0].status == "Enabled"
    error_message = "Versioning must be enabled."
  }

  assert {
    condition = alltrue([
      aws_s3_bucket_public_access_block.this.block_public_acls,
      aws_s3_bucket_public_access_block.this.block_public_policy,
      aws_s3_bucket_public_access_block.this.ignore_public_acls,
      aws_s3_bucket_public_access_block.this.restrict_public_buckets,
    ])
    error_message = "All public access must be blocked."
  }

  assert {
    condition     = one(one(aws_s3_bucket_server_side_encryption_configuration.this.rule).apply_server_side_encryption_by_default).sse_algorithm == "AES256"
    error_message = "Server-side encryption must use AES256."
  }
}

run "rejects_an_invalid_name" {
  command = plan

  variables {
    name = "Invalid_Bucket_Name"
  }

  expect_failures = [var.name]
}
