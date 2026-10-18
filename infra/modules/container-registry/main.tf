# ECR repository for a Lambda container image. A separate unit from the Lambda, so the image can be
# pushed (infra/scripts/build_recommender_image.sh) before the function that runs it is created.

resource "aws_ecr_repository" "this" {
  name                 = var.name
  image_tag_mutability = "MUTABLE" # the "current" tag moves to each new build; Lambda pins the digest
  force_delete         = var.force_delete
  tags                 = var.tags

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Old builds are only kept for a quick rollback; untagged images (replaced "current") go after a day.
resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after a day"
        selection    = { tagStatus = "untagged", countType = "sinceImagePushed", countUnit = "days", countNumber = 1 }
        action       = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep the last ${var.keep_images} builds"
        selection    = { tagStatus = "tagged", tagPrefixList = ["build-"], countType = "imageCountMoreThan", countNumber = var.keep_images }
        action       = { type = "expire" }
      },
    ]
  })
}
