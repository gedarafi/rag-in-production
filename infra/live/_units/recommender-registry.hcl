# Shared by live/aws/recommender-registry and live/local/recommender-registry: the ECR repository of the
# recommender worker's image, applied before the recommender unit pushes to it.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/container-registry"
}

inputs = {
  name         = "${local.env.name_prefix}-recommender"
  force_delete = local.env.force_destroy
}
