output "url" {
  description = "Address of the sign-in page."
  value       = "https://${aws_cloudfront_distribution.site.domain_name}"
}

output "distribution_id" {
  description = "CloudFront distribution ID, for cache invalidations."
  value       = aws_cloudfront_distribution.site.id
}

output "bucket_name" {
  description = "Bucket holding the page files."
  value       = aws_s3_bucket.site.id
}
