output "api_endpoint" {
  description = "Base URL of the API, e.g. https://abc123.execute-api.eu-north-1.amazonaws.com."
  value       = aws_apigatewayv2_api.this.api_endpoint
}

output "api_domain" {
  description = "Host name of the API (no scheme or port), used as the CloudFront origin."
  value       = regex("^https?://([^:/]+)", aws_apigatewayv2_api.this.api_endpoint)[0]
}

output "chat_function_name" {
  description = "Name of the chat Lambda."
  value       = aws_lambda_function.chat.function_name
}

output "model_id" {
  description = "Bedrock model or inference profile the chat Lambda calls."
  value       = var.model_id
}

output "model_api" {
  description = "How the chat Lambda calls the model: converse (bedrock-runtime) or openai (bedrock-mantle)."
  value       = var.model_api
}
