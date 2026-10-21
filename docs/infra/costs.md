# Monthly cost estimate

What the real AWS deployment (account `422725200891`, us-east-2) costs per month, from the resources
deployed on 2026-10-21 and us-east-2 on-demand list prices. AWS's own forecast (Cost Explorer, the
budget's forecasted alert) needs more billing history before it works for this account.

| Resource | What's deployed | Instance stopped | Instance on 24/7 |
| --- | --- | --- | --- |
| EC2 recommender | `t4g.medium` (\$0.0336/h) | \$0.00 | \$24.53 |
| Public IPv4 | 1 Elastic IP (\$0.005/h) | \$3.65 | \$3.65 |
| EBS disk | 20 GB gp3 (\$0.08/GB) | \$1.60 | \$1.60 |
| Lambda | `rag-aws-chat`, `rag-aws-pre-signup` | ≈ \$0 | ≈ \$0 |
| API Gateway, CloudFront, Cognito | chat API, web, auth | ≈ \$0 | ≈ \$0 |
| S3 | 3 small buckets (web, recommender, tfstate) | ≈ \$0.01 | ≈ \$0.01 |
| CloudWatch Logs | 4 small log groups | ≈ \$0 | ≈ \$0 |
| Bedrock | pay per token | ≈ \$0.01–1 | ≈ \$0.01–1 |
| AWS Budgets | 1 budget (the first two are free) | \$0 | \$0 |
| **Monthly total** | | **≈ \$5.30** | **≈ \$30–31** |

- Every hour the instance runs adds about \$0.034; 8 hours a day on weekdays is about \$5.50 a month.
- The Elastic IP and the disk are charged all month, stopped or not. `create_instance = false`
  removes the instance, its IP and its disk, which brings the total under \$1.
- Bedrock is the only cost that grows with use: chat traffic and evaluation runs.
- Cost Explorer API calls (`aws ce ...`) cost \$0.01 each; the Cost Explorer console is free.
