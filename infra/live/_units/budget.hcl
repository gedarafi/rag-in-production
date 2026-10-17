# Shared by live/aws/budget and live/aws-mantle/budget; the values come from each environment's env.hcl.
# Real AWS only: LocalStack does not emulate AWS Budgets (and bills nothing).

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/budget"
}

inputs = {
  name      = "${local.env.name_prefix}-monthly"
  limit_usd = local.env.budget_limit_usd
  emails    = local.env.budget_emails
}
