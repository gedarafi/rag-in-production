# Plan-only tests with a mocked provider. They need no LocalStack and no credentials.

mock_provider "aws" {}

variables {
  name = "rag-unit-recommender"
}

run "scans_and_encrypts_images" {
  command = plan

  assert {
    condition     = aws_ecr_repository.this.image_scanning_configuration[0].scan_on_push
    error_message = "Images must be scanned for vulnerabilities on push."
  }

  assert {
    condition     = aws_ecr_repository.this.encryption_configuration[0].encryption_type == "AES256"
    error_message = "Images must be encrypted at rest."
  }

  assert {
    condition     = !aws_ecr_repository.this.force_delete
    error_message = "Images must survive a destroy unless asked."
  }
}

run "expires_old_builds" {
  command = plan

  assert {
    condition     = jsondecode(aws_ecr_lifecycle_policy.this.policy).rules[1].selection.countNumber == 5
    error_message = "Only the last 5 builds are kept by default."
  }
}

run "rejects_invalid_names" {
  command = plan

  variables {
    name = "Not/Valid"
  }

  expect_failures = [var.name]
}
