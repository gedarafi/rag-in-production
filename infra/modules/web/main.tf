# Static sign-in and chat page: private S3 bucket behind CloudFront.
# CloudFront also forwards /api/* to the HTTP API, so the page and API share one origin (no CORS).

data "aws_caller_identity" "current" {}

locals {
  bucket_name = "${var.name_prefix}-web-${data.aws_caller_identity.current.account_id}"

  content_types = {
    html = "text/html; charset=utf-8"
    js   = "text/javascript; charset=utf-8"
    css  = "text/css; charset=utf-8"
    svg  = "image/svg+xml"
    png  = "image/png"
    ico  = "image/x-icon"
    json = "application/json"
    txt  = "text/plain; charset=utf-8"
  }

  # config.js is generated below; config.example.js is only for local development.
  site_files = setsubtract(fileset(var.site_dir, "**"), ["config.js", "config.example.js"])

  cognito_url = "https://cognito-idp.${var.region}.amazonaws.com"

  # IDs of AWS managed CloudFront policies. AWS documents them as fixed in every account; using
  # them directly (not looking them up by name) also works on LocalStack, which has no managed policies.
  cache_policy_caching_optimized               = "658327ea-f89d-4fab-a63d-7e88639e58f6"
  cache_policy_caching_disabled                = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
  origin_request_policy_all_viewer_except_host = "b689b0a8-53d0-40ab-baf2-68738e2966ac"
}

# ---------- Bucket ----------

resource "aws_s3_bucket" "site" {
  bucket        = local.bucket_name
  force_destroy = var.force_destroy
  tags          = var.tags
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket = aws_s3_bucket.site.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "site" {
  bucket = aws_s3_bucket.site.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_object" "site" {
  for_each = local.site_files

  bucket        = aws_s3_bucket.site.id
  key           = each.value
  source        = "${var.site_dir}/${each.value}"
  etag          = filemd5("${var.site_dir}/${each.value}")
  content_type  = lookup(local.content_types, reverse(split(".", each.value))[0], "application/octet-stream")
  cache_control = "no-cache"
}

resource "aws_s3_object" "config" {
  bucket        = aws_s3_bucket.site.id
  key           = "config.js"
  content_type  = local.content_types.js
  cache_control = "no-cache"
  content = "window.APP_CONFIG = ${jsonencode({
    region              = var.region
    userPoolClientId    = var.user_pool_client_id
    allowedEmailDomains = var.allowed_email_domains
    apiBase             = ""
  })};\n"
}

# Only this distribution can read the bucket.
resource "aws_s3_bucket_policy" "site" {
  bucket = aws_s3_bucket.site.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowCloudFrontRead"
      Effect    = "Allow"
      Principal = { Service = "cloudfront.amazonaws.com" }
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.site.arn}/*"
      Condition = {
        StringEquals = { "AWS:SourceArn" = aws_cloudfront_distribution.site.arn }
      }
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.site]
}

# ---------- CloudFront ----------

resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.name_prefix}-web"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# Strict browser security headers. The page only talks to itself and to Cognito.
resource "aws_cloudfront_response_headers_policy" "security" {
  name = "${var.name_prefix}-security-headers"

  security_headers_config {
    content_security_policy {
      override                = true
      content_security_policy = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self' ${local.cognito_url}; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'"
    }
    strict_transport_security {
      override                   = true
      access_control_max_age_sec = 63072000
      include_subdomains         = true
    }
    content_type_options {
      override = true
    }
    frame_options {
      override     = true
      frame_option = "DENY"
    }
    referrer_policy {
      override        = true
      referrer_policy = "no-referrer"
    }
  }
}

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  comment             = "${var.name_prefix} web"
  default_root_object = "index.html"
  price_class         = var.price_class
  http_version        = "http2and3"
  tags                = var.tags

  origin {
    origin_id                = "site"
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  origin {
    origin_id   = "api"
    domain_name = var.api_domain

    custom_origin_config {
      http_port              = var.api_origin_protocol_policy == "http-only" ? var.api_origin_port : 80
      https_port             = var.api_origin_protocol_policy == "http-only" ? 443 : var.api_origin_port
      origin_protocol_policy = var.api_origin_protocol_policy
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id           = "site"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = local.cache_policy_caching_optimized
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security.id
  }

  # API calls are never cached. CloudFront always forwards the Authorization header on POST.
  ordered_cache_behavior {
    path_pattern             = "/api/*"
    target_origin_id         = "api"
    viewer_protocol_policy   = "https-only"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    compress                 = true
    cache_policy_id          = local.cache_policy_caching_disabled
    origin_request_policy_id = local.origin_request_policy_all_viewer_except_host
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
