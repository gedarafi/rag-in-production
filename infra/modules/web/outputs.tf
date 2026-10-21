output "url" {
  description = "Address of the sign-in page."
  value       = "https://${var.domain_name == "" ? aws_cloudfront_distribution.site.domain_name : var.domain_name}"
}

output "cloudfront_domain" {
  description = "CloudFront address; with domain_name set, that name needs a CNAME (DNS only) pointing here."
  value       = aws_cloudfront_distribution.site.domain_name
}

output "certificate_validation_record" {
  description = "CNAME that proves ownership of domain_name to ACM; null without a custom domain."
  value = var.domain_name == "" ? null : one([
    for o in aws_acm_certificate.site[0].domain_validation_options : {
      type  = o.resource_record_type
      name  = o.resource_record_name
      value = o.resource_record_value
    }
  ])
}

output "distribution_id" {
  description = "CloudFront distribution ID, for cache invalidations."
  value       = aws_cloudfront_distribution.site.id
}

output "bucket_name" {
  description = "Bucket holding the page files."
  value       = aws_s3_bucket.site.id
}
