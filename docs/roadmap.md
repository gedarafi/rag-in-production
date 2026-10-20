# Roadmap

## Today

About $30 a month for 30 users.

- **Compute:** one on-demand EC2 `t4g.medium` (2 vCPU, 4 GB), about $25 a month.
- **Embeddings:** generated locally, stored in S3 and loaded into the instance's memory.
- **LLM:** `qwen.qwen3-32b` on Bedrock, about $0.15 per million input tokens and $0.60 per million
  output tokens.

## Next steps

About $60 a month for 30 users.

- Refactor the app and change how it reads its data.
- Generate embeddings with Amazon Bedrock Titan instead of locally.
- Move the vector index and chunk text from the instance's memory to PostgreSQL with pgvector.
- Move `catalog.sqlite3` to PostgreSQL.
- Replace EC2 with Lambda response streaming.

## To research

- Serverless PostgreSQL: Neon.
