# Shared by live/aws/auth and live/local/auth; the values come from each environment's env.hcl.

locals {
  env = read_terragrunt_config(find_in_parent_folders("env.hcl")).locals
}

terraform {
  source = "${get_repo_root()}/infra/modules/auth"
}

inputs = {
  name_prefix           = local.env.name_prefix
  allowed_email_domains = local.env.allowed_email_domains
  ses_identity          = local.env.ses_identity
  route53_zone_id       = local.env.route53_zone_id
  create_ses_identity   = local.env.create_ses_identity
  deletion_protection   = local.env.deletion_protection
}
