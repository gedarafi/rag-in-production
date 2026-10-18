# Shared by live/aws/vector-db and live/local/vector-db: the Aurora PostgreSQL cluster (pgvector) that holds
# the recommender's passage index. LocalStack emulates it locally: RDS, the Data API and pgvector.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/vector-db"
}

inputs = {
  name = "${local.env.name_prefix}-vectors"
}
