output "cluster_arn" {
  description = "ARN of the cluster: the Data API's resourceArn."
  value       = aws_rds_cluster.this.arn
}

output "secret_arn" {
  description = "ARN of the master user's secret in Secrets Manager: the Data API's secretArn."
  value       = aws_rds_cluster.this.master_user_secret[0].secret_arn
}

output "database" {
  description = "Database the passage tables live in."
  value       = aws_rds_cluster.this.database_name
}
