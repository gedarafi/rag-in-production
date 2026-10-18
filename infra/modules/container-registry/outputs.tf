output "repository_name" {
  description = "Name of the repository."
  value       = aws_ecr_repository.this.name
}

output "repository_url" {
  description = "Registry and repository to push to, e.g. 123456789012.dkr.ecr.us-east-2.amazonaws.com/rag-aws-recommender."
  value       = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  description = "ARN of the repository."
  value       = aws_ecr_repository.this.arn
}
