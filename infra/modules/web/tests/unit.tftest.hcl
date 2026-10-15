# Plan-only tests with a mocked provider. They need no LocalStack and no credentials.

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }
  mock_resource "aws_s3_bucket" {
    defaults = {
      arn = "arn:aws:s3:::rag-unit-web-123456789012"
    }
  }
  mock_resource "aws_cloudfront_distribution" {
    defaults = {
      arn         = "arn:aws:cloudfront::123456789012:distribution/E123"
      domain_name = "d111111abcdef8.cloudfront.net"
    }
  }
}

variables {
  name_prefix           = "rag-unit"
  site_dir              = "../../../web"
  region                = "eu-north-1"
  user_pool_client_id   = "client123"
  allowed_email_domains = ["uu.se"]
  api_domain            = "abc123.execute-api.eu-north-1.amazonaws.com"
}

run "uploads_the_page_and_generates_config" {
  command = plan

  assert {
    condition     = contains(keys(aws_s3_object.site), "index.html") && contains(keys(aws_s3_object.site), "app.js")
    error_message = "The page files must be uploaded."
  }

  assert {
    condition     = !contains(keys(aws_s3_object.site), "config.example.js") && !contains(keys(aws_s3_object.site), "config.js")
    error_message = "Local config files must not be uploaded; config.js is generated."
  }

  assert {
    condition     = aws_s3_object.site["index.html"].content_type == "text/html; charset=utf-8"
    error_message = "HTML must be served as text/html."
  }

  assert {
    condition     = strcontains(aws_s3_object.config.content, "\"userPoolClientId\":\"client123\"") && strcontains(aws_s3_object.config.content, "\"apiBase\":\"\"")
    error_message = "config.js must carry the client ID and call the API on the same origin."
  }
}

run "keeps_the_bucket_private" {
  command = plan

  assert {
    condition = alltrue([
      aws_s3_bucket_public_access_block.site.block_public_acls,
      aws_s3_bucket_public_access_block.site.block_public_policy,
      aws_s3_bucket_public_access_block.site.ignore_public_acls,
      aws_s3_bucket_public_access_block.site.restrict_public_buckets,
    ])
    error_message = "All public access must be blocked; only CloudFront reads the bucket."
  }

  assert {
    condition     = aws_s3_bucket.site.bucket == "rag-unit-web-123456789012"
    error_message = "The bucket name must include the account ID to be globally unique."
  }
}

run "routes_api_calls_without_caching" {
  command = apply

  assert {
    condition     = one([for b in aws_cloudfront_distribution.site.ordered_cache_behavior : b.target_origin_id if b.path_pattern == "/api/*"]) == "api"
    error_message = "/api/* must go to the API origin."
  }

  assert {
    condition     = strcontains(aws_cloudfront_response_headers_policy.security.security_headers_config[0].content_security_policy[0].content_security_policy, "connect-src 'self' https://cognito-idp.eu-north-1.amazonaws.com")
    error_message = "The CSP must only allow the page itself and Cognito."
  }

  assert {
    condition     = output.url == "https://d111111abcdef8.cloudfront.net"
    error_message = "url must be the CloudFront address."
  }
}

run "reaches_a_localstack_api_over_http" {
  command = plan

  variables {
    api_origin_port            = 4566
    api_origin_protocol_policy = "http-only"
  }

  assert {
    condition     = one([for o in aws_cloudfront_distribution.site.origin : o.custom_origin_config[0].http_port if o.origin_id == "api"]) == 4566
    error_message = "The API origin must use the given port over HTTP."
  }
}
