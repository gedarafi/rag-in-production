output "bucket_name" {
  description = "Data bucket: publish the library to recommender/ and the wheel to releases/."
  value       = module.data.bucket_name
}

output "instance_id" {
  description = "EC2 instance ID (for SSM, start and stop); empty when create_instance is false."
  value       = var.create_instance ? aws_instance.host[0].id : ""
}

output "origin_domain" {
  description = "Stable public DNS name of the instance, the CloudFront origin for /recommend/*."
  value       = var.create_instance ? aws_eip.host[0].public_dns : ""
}

output "origin_port" {
  description = "Port of the service on the origin."
  value       = var.port
}
