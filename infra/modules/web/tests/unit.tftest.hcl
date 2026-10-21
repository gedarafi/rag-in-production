# Plan-only tests with a mocked provider. They need no LocalStack and no credentials.

mock_provider "aws" {
  alias = "us_east_1"

  mock_resource "aws_acm_certificate" {
    defaults = {
      arn = "arn:aws:acm:us-east-1:123456789012:certificate/abc"
    }
  }
}

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

run "leaves_out_the_recommender_route_by_default" {
  command = plan

  assert {
    condition     = length([for b in aws_cloudfront_distribution.site.ordered_cache_behavior : b if b.path_pattern == "/recommend/*"]) == 0
    error_message = "Without a recommender origin there must be no /recommend/* route."
  }
}

run "routes_recommender_calls_to_the_instance_without_caching" {
  command = plan

  variables {
    recommender_origin_domain = "ec2-3-1-2-3.us-east-2.compute.amazonaws.com"
  }

  assert {
    condition     = one([for b in aws_cloudfront_distribution.site.ordered_cache_behavior : b if b.path_pattern == "/recommend/*"]).target_origin_id == "recommender" && one([for b in aws_cloudfront_distribution.site.ordered_cache_behavior : b if b.path_pattern == "/recommend/*"]).cache_policy_id == "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    error_message = "/recommend/* must go to the recommender origin, uncached."
  }

  assert {
    condition     = one([for o in aws_cloudfront_distribution.site.origin : o if o.origin_id == "recommender"]).custom_origin_config[0].http_port == 8080
    error_message = "CloudFront must reach the service on its port."
  }

  assert {
    condition     = strcontains(aws_s3_object.config.content, "\"recommender\":true")
    error_message = "The page must know the recommender is available."
  }
}

run "serves_a_custom_domain_with_its_us_east_1_certificate" {
  command = plan

  variables {
    domain_name = "rag.example.com"
  }

  assert {
    condition     = aws_cloudfront_distribution.site.aliases == toset(["rag.example.com"]) && aws_acm_certificate.site[0].domain_name == "rag.example.com"
    error_message = "The distribution must answer on the custom domain, with a certificate for it."
  }

  assert {
    condition     = aws_cloudfront_distribution.site.viewer_certificate[0].ssl_support_method == "sni-only" && !aws_cloudfront_distribution.site.viewer_certificate[0].cloudfront_default_certificate
    error_message = "With a custom domain the distribution must use its own certificate over SNI."
  }

  assert {
    condition     = output.url == "https://rag.example.com"
    error_message = "url must be the custom domain."
  }
}
