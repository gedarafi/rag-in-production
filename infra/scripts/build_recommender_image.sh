#!/usr/bin/env bash
# Build the recommender worker's container image (infra/docker/recommender.Dockerfile) and push it to
# its ECR repository (the recommender-registry unit), tagged build-<content hash> and "current". It then
# writes the pushed digest to image.auto.tfvars in the current directory (Terragrunt runs hooks in the
# unit's working copy of the module), so the recommender module deploys exactly that image. Terragrunt
# runs it before every plan and apply of the recommender unit; Docker's layer cache makes an unchanged
# build a no-op push.
#
# Usage: build_recommender_image.sh <repository name> <region> <localstack|aws>
set -euo pipefail
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
name="${1:?usage: build_recommender_image.sh <repository name> <region> <localstack|aws>}"
region="${2:?region}"
target="${3:?localstack or aws}"

aws_cli() {
  if [ "$target" = localstack ]; then
    AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test aws --endpoint-url "${LOCALSTACK_ENDPOINT:-http://localhost:4566}" --region "$region" "$@"
  else
    aws --region "$region" "$@"
  fi
}
if ! repository="$(aws_cli ecr describe-repositories --repository-names "$name" \
    --query 'repositories[0].repositoryUri' --output text 2>/dev/null)"; then
  echo "ECR repository $name doesn't exist yet (apply recommender-registry first); skipping the image build." >&2
  exit 0
fi

out="$root/infra/.build/recommender"
mkdir -p "$out"
# The recommender dependency group, pinned by uv.lock.
uv export --quiet --project "$root" --frozen --only-group recommender --no-hashes --no-header --no-annotate \
  --no-emit-project > "$out/requirements.txt"

# The tag changes exactly when something that goes into the image changes.
hash="$(cd "$root" && { cat infra/docker/recommender.Dockerfile infra/.build/recommender/requirements.txt \
  src/__init__.py src/tracing.py src/chat/__init__.py src/chat/store.py;
  find src/recommender -name '*.py' -print0 | sort -z | xargs -0 cat; } | sha256sum | cut -c1-16)"

if [ "$target" = aws ]; then
  aws_cli ecr get-login-password | docker login --username AWS --password-stdin "${repository%%/*}" >/dev/null
fi

# Lambda needs a single-platform image manifest: no attestations.
docker build --quiet --platform linux/amd64 --provenance=false \
  -f "$root/infra/docker/recommender.Dockerfile" -t "$repository:build-$hash" -t "$repository:current" "$root" >/dev/null
docker push --quiet "$repository:build-$hash" >/dev/null
docker push --quiet "$repository:current" >/dev/null
digest="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$repository:current" \
  | grep -F "$repository@" | head -1 | cut -d@ -f2)"
[[ "$digest" == sha256:* ]] || { echo "Could not read the digest of $repository:current." >&2; exit 1; }
printf 'image_digest = "%s"\n' "$digest" > image.auto.tfvars
echo "Pushed $repository:build-$hash ($digest) as current." >&2
