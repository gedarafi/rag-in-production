include "root" {
  path = find_in_parent_folders("root.hcl")
}

include "unit" {
  path = "${get_repo_root()}/infra/live/_units/auth.hcl"
}
