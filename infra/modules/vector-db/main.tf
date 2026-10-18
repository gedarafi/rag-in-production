# Aurora Serverless v2 PostgreSQL with pgvector: the recommender's passage index. The worker queries it
# through the RDS Data API (HTTPS, authorised by IAM and the master user's secret), so no Lambda joins a
# VPC and nothing needs a NAT gateway. The cluster pauses at 0 ACU when idle; the first query after a
# pause waits for it to resume (about 15 s). Its content is derived from published bundles, so a
# destroyed cluster is rebuilt by publishing again (`just recommender-publish`).

# The cluster needs subnets in two availability zones. Only it lives in this VPC: no internet gateway,
# no NAT, no endpoints.
data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "this" {
  cidr_block = var.cidr_block
  tags       = merge(var.tags, { Name = var.name })
}

resource "aws_subnet" "this" {
  count             = 2
  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.cidr_block, 1, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = merge(var.tags, { Name = "${var.name}-${count.index + 1}" })
}

resource "aws_db_subnet_group" "this" {
  name       = var.name
  subnet_ids = aws_subnet.this[*].id
  tags       = var.tags
}

resource "aws_rds_cluster" "this" {
  cluster_identifier          = var.name
  engine                      = "aurora-postgresql"
  engine_mode                 = "provisioned" # Serverless v2 is the provisioned mode with db.serverless instances
  engine_version              = var.engine_version
  database_name               = var.database
  master_username             = "vectors"
  manage_master_user_password = true # RDS keeps the password in Secrets Manager; the Data API signs in with it
  enable_http_endpoint        = true # the RDS Data API
  db_subnet_group_name        = aws_db_subnet_group.this.name
  storage_encrypted           = true
  deletion_protection         = var.deletion_protection
  skip_final_snapshot         = true # derived data: publishing a bundle loads it again
  backup_retention_period     = 1
  tags                        = var.tags

  serverlessv2_scaling_configuration {
    min_capacity             = 0
    max_capacity             = var.max_capacity
    seconds_until_auto_pause = var.seconds_until_auto_pause
  }

  lifecycle {
    # Aurora applies minor versions itself; a plan must not try to go back.
    ignore_changes = [engine_version]
  }
}

resource "aws_rds_cluster_instance" "this" {
  identifier         = "${var.name}-1"
  cluster_identifier = aws_rds_cluster.this.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.this.engine
  engine_version     = aws_rds_cluster.this.engine_version
  tags               = var.tags

  lifecycle {
    ignore_changes = [engine_version]
  }
}
