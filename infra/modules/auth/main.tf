# Passwordless sign-in: users register and sign in with a one-time code sent to their email.
# Only addresses in var.allowed_email_domains can register (pre sign-up Lambda).

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  identity_is_domain = !strcontains(var.ses_identity, "@")
  from_address       = local.identity_is_domain ? "no-reply@${var.ses_identity}" : var.ses_identity
  pre_signup_name    = "${var.name_prefix}-pre-signup"

  # When the identity is managed elsewhere (or on LocalStack, which can't tag SESv2 identities), point at it by ARN.
  ses_identity_arn = var.create_ses_identity ? aws_sesv2_email_identity.sender[0].arn : "arn:aws:ses:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:identity/${var.ses_identity}"
}

# ---------- SES: Cognito sends the codes through your own identity ----------
# Email one-time codes require the Essentials tier and an SES (DEVELOPER) email configuration.

resource "aws_sesv2_email_identity" "sender" {
  count = var.create_ses_identity ? 1 : 0

  email_identity = var.ses_identity
  tags           = var.tags

  # Easy DKIM with 2048-bit keys (only used when the identity is a domain).
  dynamic "dkim_signing_attributes" {
    for_each = local.identity_is_domain ? [1] : []
    content {
      next_signing_key_length = "RSA_2048_BIT"
    }
  }
}

# Easy DKIM records prove domain ownership and keep the codes out of spam folders.
resource "aws_route53_record" "dkim" {
  count = var.create_ses_identity && local.identity_is_domain && var.route53_zone_id != null ? 3 : 0

  zone_id = var.route53_zone_id
  name    = "${aws_sesv2_email_identity.sender[0].dkim_signing_attributes[0].tokens[count.index]}._domainkey.${var.ses_identity}"
  type    = "CNAME"
  ttl     = 600
  records = ["${aws_sesv2_email_identity.sender[0].dkim_signing_attributes[0].tokens[count.index]}.dkim.amazonses.com"]
}

resource "aws_route53_record" "dmarc" {
  count = var.create_ses_identity && local.identity_is_domain && var.route53_zone_id != null ? 1 : 0

  zone_id = var.route53_zone_id
  name    = "_dmarc.${var.ses_identity}"
  type    = "TXT"
  ttl     = 600
  records = ["v=DMARC1; p=none;"]
}

# ---------- Pre sign-up Lambda: the email-domain gate ----------

data "archive_file" "pre_signup" {
  type        = "zip"
  source_file = "${var.lambda_source_dir}/pre_signup.py"
  output_path = "${path.module}/build/pre_signup.zip"
}

resource "aws_iam_role" "pre_signup" {
  name = local.pre_signup_name
  tags = var.tags

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "pre_signup_logs" {
  role       = aws_iam_role.pre_signup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Created before the function so Lambda never makes an unmanaged log group with no retention.
resource "aws_cloudwatch_log_group" "pre_signup" {
  name              = "/aws/lambda/${local.pre_signup_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_lambda_function" "pre_signup" {
  function_name    = local.pre_signup_name
  role             = aws_iam_role.pre_signup.arn
  runtime          = "python3.13"
  handler          = "pre_signup.handler"
  filename         = data.archive_file.pre_signup.output_path
  source_code_hash = data.archive_file.pre_signup.output_base64sha256
  timeout          = 5
  memory_size      = 128
  tags             = var.tags

  environment {
    variables = {
      ALLOWED_DOMAINS = join(",", var.allowed_email_domains)
    }
  }

  depends_on = [aws_cloudwatch_log_group.pre_signup, aws_iam_role_policy_attachment.pre_signup_logs]
}

# ---------- User pool ----------

resource "aws_cognito_user_pool" "this" {
  name                = "${var.name_prefix}-users"
  user_pool_tier      = "ESSENTIALS"
  deletion_protection = var.deletion_protection
  tags                = var.tags

  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  username_configuration {
    case_sensitive = false
  }

  # PASSWORD has to stay allowed at pool level; the app client only offers USER_AUTH and
  # users are created without a password, so in practice everyone signs in with EMAIL_OTP.
  sign_in_policy {
    allowed_first_auth_factors = ["PASSWORD", "EMAIL_OTP"]
  }

  email_configuration {
    email_sending_account = "DEVELOPER"
    from_email_address    = "${var.from_name} <${local.from_address}>"
    source_arn            = local.ses_identity_arn
  }

  verification_message_template {
    default_email_option = "CONFIRM_WITH_CODE"
    email_subject        = "Your ${var.from_name} code"
    email_message        = "Your verification code is {####}. It expires in 24 hours."
  }

  lambda_config {
    pre_sign_up = aws_lambda_function.pre_signup.arn
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  admin_create_user_config {
    allow_admin_create_user_only = false
  }
}

resource "aws_lambda_permission" "cognito_pre_signup" {
  statement_id  = "AllowCognitoInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pre_signup.function_name
  principal     = "cognito-idp.amazonaws.com"
  source_arn    = aws_cognito_user_pool.this.arn
}

# ---------- App client for the browser (public, no secret) ----------

resource "aws_cognito_user_pool_client" "web" {
  name         = "${var.name_prefix}-web"
  user_pool_id = aws_cognito_user_pool.this.id

  generate_secret               = false
  prevent_user_existence_errors = "ENABLED"
  enable_token_revocation       = true

  # USER_AUTH is choice-based sign-in, where EMAIL_OTP lives. No password or SRP flows.
  explicit_auth_flows = ["ALLOW_USER_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"]

  access_token_validity  = 1
  id_token_validity      = 1
  refresh_token_validity = 30

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }
}
