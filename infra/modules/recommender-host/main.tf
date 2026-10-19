# The recommender on one EC2 instance: it pulls the published library from the data bucket, calls
# Bedrock for every LLM step, and answers the web app's signed-in users through CloudFront.
#
#   data bucket   recommender/<version>/  the library (`recommender publish`, src/recommender/bundle.py)
#                 releases/<wheel>        the Python package the instance installs
#   instance      systemd runs `recommender serve`; access is through SSM Session Manager, no SSH
#   Elastic IP    a stable origin for CloudFront across stop/start

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  name   = "${var.name_prefix}-recommender"
  bucket = "${local.name}-${data.aws_caller_identity.current.account_id}"
  arm    = can(regex("^[a-z][0-9]+g[a-z]*\\.", var.instance_type))
  ami_id = var.ami_id != "" ? var.ami_id : one(data.aws_ssm_parameter.ami[*].value)
}

module "data" {
  source = "../s3-bucket"

  name          = local.bucket
  force_destroy = var.force_destroy
  tags          = var.tags
}

# ---------- Instance role: read the library, call Bedrock, be reachable through SSM ----------

resource "aws_iam_role" "host" {
  name = local.name
  tags = var.tags

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "host" {
  name = local.name
  role = aws_iam_role.host.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = ["${module.data.bucket_arn}/recommender/*", "${module.data.bucket_arn}/releases/*"]
      },
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = module.data.bucket_arn
      },
      {
        # bedrock-mantle authorizes the project a request runs in: the account's default project.
        Effect   = "Allow"
        Action   = "bedrock-mantle:CreateInference"
        Resource = "arn:aws:bedrock-mantle:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:project/default"
      },
    ]
  })
}

# The tracing API key (Arize AX) stays in SSM, out of the user data and the state; the service reads
# it at start. The default aws/ssm key needs no KMS grant.
resource "aws_iam_role_policy" "tracing_key" {
  count = var.tracing_api_key_parameter == "" ? 0 : 1
  name  = "read-tracing-key"
  role  = aws_iam_role.host.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = "arn:aws:ssm:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:parameter/${trimprefix(var.tracing_api_key_parameter, "/")}"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.host.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "host" {
  name = local.name
  role = aws_iam_role.host.name
  tags = var.tags
}

# ---------- Network: only CloudFront reaches the service ----------

data "aws_vpc" "default" {
  default = true
}

data "aws_ec2_managed_prefix_list" "cloudfront" {
  count = length(var.ingress_cidr_blocks) == 0 ? 1 : 0
  name  = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_security_group" "host" {
  name        = local.name
  description = "Recommender service: HTTP from CloudFront only"
  vpc_id      = data.aws_vpc.default.id
  tags        = var.tags

  ingress {
    description     = "Recommender HTTP"
    from_port       = var.port
    to_port         = var.port
    protocol        = "tcp"
    prefix_list_ids = length(var.ingress_cidr_blocks) == 0 ? [data.aws_ec2_managed_prefix_list.cloudfront[0].id] : null
    cidr_blocks     = length(var.ingress_cidr_blocks) == 0 ? null : var.ingress_cidr_blocks
  }

  # Bedrock, S3, SSM, the package index and the embedding model download.
  egress {
    description = "HTTPS out"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "HTTP out (package mirrors)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------- Instance ----------

data "aws_ssm_parameter" "ami" {
  count = var.create_instance && var.ami_id == "" ? 1 : 0
  name  = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-${local.arm ? "arm64" : "x86_64"}"
}

resource "aws_instance" "host" {
  count = var.create_instance ? 1 : 0

  ami                    = local.ami_id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.host.name
  vpc_security_group_ids = [aws_security_group.host.id]
  tags                   = merge(var.tags, { Name = local.name })

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    bucket                    = local.bucket
    region                    = data.aws_region.current.region
    wheel                     = var.wheel
    port                      = var.port
    workers                   = var.workers
    cognito_issuer            = var.cognito_issuer
    cognito_client            = var.cognito_client_id
    bedrock_model             = var.bedrock_model
    tracing_endpoint          = var.tracing_endpoint
    tracing_project           = var.tracing_project
    tracing_space_id          = var.tracing_space_id
    tracing_api_key_parameter = var.tracing_api_key_parameter
  })

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  lifecycle {
    ignore_changes = [ami]
  }

  # The boot script reads the wheel from the bucket.
  depends_on = [module.data]
}

resource "aws_eip" "host" {
  count    = var.create_instance ? 1 : 0
  domain   = "vpc"
  instance = aws_instance.host[0].id
  tags     = merge(var.tags, { Name = local.name })
}
