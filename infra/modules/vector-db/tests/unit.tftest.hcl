# Plan-only tests with a mocked provider. They need no LocalStack and no credentials.

mock_provider "aws" {
  mock_data "aws_availability_zones" {
    defaults = {
      names = ["us-east-2a", "us-east-2b", "us-east-2c"]
    }
  }
  mock_resource "aws_rds_cluster" {
    defaults = {
      arn = "arn:aws:rds:us-east-2:123456789012:cluster:rag-unit-vectors"
      master_user_secret = [{
        secret_arn    = "arn:aws:secretsmanager:us-east-2:123456789012:secret:rds-rag-unit-vectors"
        kms_key_id    = ""
        secret_status = "active"
      }]
    }
  }
}

variables {
  name = "rag-unit-vectors"
}

run "pauses_when_idle_and_serves_the_data_api" {
  command = plan

  assert {
    condition     = aws_rds_cluster.this.serverlessv2_scaling_configuration[0].min_capacity == 0 && aws_rds_cluster.this.serverlessv2_scaling_configuration[0].seconds_until_auto_pause == 300
    error_message = "The cluster must scale to 0 ACU after five idle minutes."
  }

  assert {
    condition     = aws_rds_cluster.this.enable_http_endpoint && aws_rds_cluster.this.manage_master_user_password
    error_message = "The worker queries through the Data API, which signs in with the managed master secret."
  }

  assert {
    condition     = aws_rds_cluster_instance.this.instance_class == "db.serverless"
    error_message = "The instance must be Serverless v2."
  }

  assert {
    condition     = aws_rds_cluster.this.storage_encrypted
    error_message = "Storage must be encrypted at rest."
  }
}

run "subnets_in_two_zones_without_internet" {
  command = plan

  assert {
    condition     = length(toset(aws_subnet.this[*].availability_zone)) == 2
    error_message = "Aurora needs subnets in two availability zones."
  }

  assert {
    condition     = !anytrue(aws_subnet.this[*].map_public_ip_on_launch)
    error_message = "The cluster's subnets must be private."
  }
}

run "exports_what_the_data_api_needs" {
  command = apply

  assert {
    condition     = output.secret_arn == "arn:aws:secretsmanager:us-east-2:123456789012:secret:rds-rag-unit-vectors" && output.database == "vectors"
    error_message = "The worker needs the cluster ARN, the secret ARN and the database."
  }
}

run "rejects_a_pause_below_five_minutes" {
  command = plan

  variables {
    seconds_until_auto_pause = 60
  }

  expect_failures = [var.seconds_until_auto_pause]
}
