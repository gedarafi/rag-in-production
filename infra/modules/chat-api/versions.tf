terraform {
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # 6.69.1 fixes "bufio.Scanner: token too long" when uploading a Lambda zip of more than 64 KB.
      version = ">= 6.69.1, < 7.0.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.7"
    }
  }
}
