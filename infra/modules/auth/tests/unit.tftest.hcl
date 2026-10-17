# Plan-only tests with mocked providers. They need no LocalStack and no credentials.

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }
  mock_data "aws_region" {
    defaults = {
      region = "eu-north-1"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/rag-unit-pre-signup"
    }
  }
  mock_resource "aws_lambda_function" {
    defaults = {
      arn = "arn:aws:lambda:eu-north-1:123456789012:function:rag-unit-pre-signup"
    }
  }
  mock_resource "aws_cognito_user_pool" {
    defaults = {
      arn      = "arn:aws:cognito-idp:eu-north-1:123456789012:userpool/eu-north-1_TEST"
      endpoint = "cognito-idp.eu-north-1.amazonaws.com/eu-north-1_TEST"
    }
  }
  mock_resource "aws_sesv2_email_identity" {
    defaults = {
      arn = "arn:aws:ses:eu-north-1:123456789012:identity/uurag.se"
      dkim_signing_attributes = {
        tokens = ["tok1", "tok2", "tok3"]
      }
    }
  }
}

mock_provider "archive" {}

variables {
  name_prefix         = "rag-unit"
  lambda_source_dir   = "../../../src/auth"
  ses_identity        = "uurag.se"
  route53_zone_id     = "Z123EXAMPLE"
  deletion_protection = "INACTIVE"
  tags                = { environment = "test" }
}

run "configures_passwordless_email_sign_in" {
  command = plan

  assert {
    condition     = aws_cognito_user_pool.this.user_pool_tier == "ESSENTIALS"
    error_message = "Email one-time codes need the Essentials tier."
  }

  assert {
    condition     = contains(aws_cognito_user_pool.this.sign_in_policy[0].allowed_first_auth_factors, "EMAIL_OTP")
    error_message = "EMAIL_OTP must be an allowed first factor."
  }

  assert {
    condition     = aws_cognito_user_pool.this.email_configuration[0].email_sending_account == "DEVELOPER"
    error_message = "Email one-time codes must be sent through SES."
  }

  assert {
    condition     = aws_cognito_user_pool.this.email_configuration[0].from_email_address == "RAG in Production <no-reply@uurag.se>"
    error_message = "A domain identity must send from no-reply@<domain>."
  }

  assert {
    condition     = toset(aws_cognito_user_pool_client.web.explicit_auth_flows) == toset(["ALLOW_USER_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"])
    error_message = "The web client must only allow choice-based auth and refresh, no password flows."
  }

  assert {
    condition     = aws_cognito_user_pool_client.web.generate_secret == false
    error_message = "A browser client cannot keep a secret."
  }
}

run "gates_registration_by_email_domain" {
  command = plan

  assert {
    condition     = aws_lambda_function.pre_signup.environment[0].variables["ALLOWED_DOMAINS"] == "uu.se,student.uu.se"
    error_message = "The pre sign-up Lambda must receive the allowed domains."
  }

  assert {
    condition     = aws_cognito_user_pool.this.lambda_config[0].pre_sign_up == aws_lambda_function.pre_signup.arn
    error_message = "The user pool must call the pre sign-up Lambda."
  }
}

run "creates_dns_records_for_a_domain_identity" {
  command = apply

  assert {
    condition     = length(aws_route53_record.dkim) == 3 && length(aws_route53_record.dmarc) == 1
    error_message = "A domain identity with a hosted zone needs three DKIM records and DMARC."
  }

  assert {
    condition     = aws_route53_record.dkim[0].records == toset(["tok1.dkim.amazonses.com"])
    error_message = "DKIM records must point at the SES tokens."
  }
}

run "email_address_identity_needs_no_dns" {
  command = plan

  variables {
    ses_identity    = "me@student.uu.se"
    route53_zone_id = null
  }

  # An email identity has no DKIM block, so drop the mocked DKIM tokens.
  override_resource {
    target = aws_sesv2_email_identity.sender
    values = {
      arn = "arn:aws:ses:eu-north-1:123456789012:identity/me@student.uu.se"
    }
  }

  assert {
    condition     = length(aws_route53_record.dkim) == 0 && length(aws_route53_record.dmarc) == 0
    error_message = "An email address identity must not create DNS records."
  }

  assert {
    condition     = aws_cognito_user_pool.this.email_configuration[0].from_email_address == "RAG in Production <me@student.uu.se>"
    error_message = "An email identity must send from that address."
  }
}

run "uses_an_existing_identity_by_arn" {
  command = plan

  variables {
    ses_identity        = "no-reply@example.com"
    route53_zone_id     = null
    create_ses_identity = false
  }

  assert {
    condition     = length(aws_sesv2_email_identity.sender) == 0
    error_message = "No SES identity must be created when create_ses_identity is false."
  }

  assert {
    condition     = aws_cognito_user_pool.this.email_configuration[0].source_arn == "arn:aws:ses:eu-north-1:123456789012:identity/no-reply@example.com"
    error_message = "Cognito must send through the existing identity's ARN."
  }
}

run "rejects_domains_with_at_sign" {
  command = plan

  variables {
    allowed_email_domains = ["@uu.se"]
  }

  expect_failures = [var.allowed_email_domains]
}
