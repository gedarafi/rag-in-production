# Current implementation

Aprox 30 USD for 30 Users.

## Compute

EC2 on Demand t4g.medium

* 4 GB
* 2 vCPU
* 20 USD/Month

## Embeddings

* Loaded in the EC2.
* Stored on S3
* Generated local

## LLM

* qwen.qwen3-32b: about $0.15 per million input tokens, $0.60 per million output.

# Next steps

* Refactor app, change the way information is consume.
* Move embeddings from S3,EC2 to Amazon Bedrock Titan embeddings.
* Move index+text.md from EC2(memory) to Amazon RDS for PostgreSQL with pgvector extension.
* Move catalog.sqlite3 to postgres.
* Move EC2 to Lambda streaming.

Aprox 60 USD for 30 Users.

# To research

* Serverless postgresql - Neon.