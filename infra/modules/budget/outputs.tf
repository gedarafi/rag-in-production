output "budget_name" {
  description = "Name of the budget."
  value       = aws_budgets_budget.this.name
}

output "limit_usd" {
  description = "Monthly cost limit in USD."
  value       = var.limit_usd
}
