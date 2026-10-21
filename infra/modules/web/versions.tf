terraform {
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.66"
      # The custom domain's certificate lives in us-east-1, as CloudFront requires.
      configuration_aliases = [aws.us_east_1]
    }
  }
}
