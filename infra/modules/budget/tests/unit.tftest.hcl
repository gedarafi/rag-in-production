# Plan-only tests with a mocked AWS provider. They need no credentials.
# There is no LocalStack test: LocalStack does not emulate AWS Budgets.

mock_provider "aws" {}

variables {
  name      = "rag-unit-test-monthly"
  limit_usd = 10
  emails    = ["me@example.com"]
}

run "configures_a_monthly_cost_budget" {
  command = plan

  assert {
    condition     = aws_budgets_budget.this.name == "rag-unit-test-monthly"
    error_message = "The budget must use the given name."
  }

  assert {
    condition     = aws_budgets_budget.this.budget_type == "COST" && aws_budgets_budget.this.time_unit == "MONTHLY"
    error_message = "The budget must track cost per month."
  }

  assert {
    condition     = tonumber(aws_budgets_budget.this.limit_amount) == 10 && aws_budgets_budget.this.limit_unit == "USD"
    error_message = "The limit must be the given amount in USD."
  }
}

run "emails_on_actual_80_and_forecast_100_percent" {
  command = plan

  assert {
    condition = toset([for n in aws_budgets_budget.this.notification : "${n.notification_type}:${n.threshold}:${n.threshold_type}"]) == toset([
      "ACTUAL:80:PERCENTAGE",
      "FORECASTED:100:PERCENTAGE",
    ])
    error_message = "The budget must alert on 80 % actual and 100 % forecasted spend."
  }

  assert {
    condition     = alltrue([for n in aws_budgets_budget.this.notification : toset(n.subscriber_email_addresses) == toset(["me@example.com"])])
    error_message = "Every alert must go to the given emails."
  }
}

run "rejects_a_zero_limit" {
  command = plan

  variables {
    limit_usd = 0
  }

  expect_failures = [var.limit_usd]
}

run "rejects_no_emails" {
  command = plan

  variables {
    emails = []
  }

  expect_failures = [var.emails]
}

run "rejects_a_domain_instead_of_an_email" {
  command = plan

  variables {
    emails = ["uurag.se"]
  }

  expect_failures = [var.emails]
}
