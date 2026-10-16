output "user_pool_id" {
  description = "Cognito user pool ID."
  value       = aws_cognito_user_pool.this.id
}

output "user_pool_arn" {
  description = "Cognito user pool ARN."
  value       = aws_cognito_user_pool.this.arn
}

output "user_pool_client_id" {
  description = "Public app client ID used by the web page and as the API audience."
  value       = aws_cognito_user_pool_client.web.id
}

output "issuer" {
  description = "Token issuer URL, for the API Gateway JWT authorizer."
  value       = "https://${aws_cognito_user_pool.this.endpoint}"
}

output "ses_identity_status" {
  description = "Whether SES has verified the sending identity. Codes are only sent once this is SUCCESS."
  value       = var.create_ses_identity ? aws_sesv2_email_identity.sender[0].verification_status : "NOT_MANAGED"
}

output "ses_identity" {
  description = "Domain or email address the one-time codes are sent from."
  value       = var.ses_identity
}
