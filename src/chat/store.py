"""Chat jobs and conversations in DynamoDB, shared by the api Lambda and the recommender worker.

jobs (JOBS_TABLE), key job_id: one question and its answer, from "pending" through "running" to
"done" or "failed". The page polls it. Items expire after a day (TTL attribute expires_at).

conversations (CONVERSATIONS_TABLE), key pk = "<user>#<conversation>" and turn (number): one
question and its answer per turn, so a follow-up sees the earlier turns. The user is part of the key,
so nobody can read or extend another user's conversation. Items expire after 30 days.
"""

from __future__ import annotations

import os
import re
import time
import uuid

JOB_TTL_SECONDS = 24 * 3600
CONVERSATION_TTL_SECONDS = 30 * 24 * 3600
# The worker reads at most this many earlier turns; the pipeline trims them further to its prompt budget.
HISTORY_TURNS = 6
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(value) -> bool:
    return isinstance(value, str) and bool(ID_PATTERN.match(value))


def _item(values: dict) -> dict:
    """Plain values as DynamoDB attribute values (strings and numbers only)."""
    return {key: {"N": str(value)} if isinstance(value, (int, float)) else {"S": value}
            for key, value in values.items() if value is not None}


def _plain(item: dict) -> dict:
    return {key: int(value["N"]) if "N" in value else value["S"] for key, value in item.items()}


class ChatStore:
    def __init__(self, jobs_table: str, conversations_table: str, client=None):
        self.jobs_table, self.conversations_table = jobs_table, conversations_table
        self._client = client

    @classmethod
    def from_env(cls) -> "ChatStore":
        return cls(os.environ["JOBS_TABLE"], os.environ["CONVERSATIONS_TABLE"])

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client("dynamodb", config=Config(connect_timeout=2, read_timeout=5,
                                                                   retries={"max_attempts": 3, "mode": "standard"}))
        return self._client

    # --- Jobs ---

    def create_job(self, job_id: str, user_id: str, conversation_id: str, message: str, trace_id: str = "") -> dict:
        now = int(time.time())
        job = {"job_id": job_id, "user_id": user_id, "conversation_id": conversation_id, "message": message,
               "status": "pending", "trace_id": trace_id or None, "created_at": now, "updated_at": now,
               "expires_at": now + JOB_TTL_SECONDS}
        self.client.put_item(TableName=self.jobs_table, Item=_item(job),
                             ConditionExpression="attribute_not_exists(job_id)")
        return job

    def get_job(self, job_id: str) -> dict | None:
        item = self.client.get_item(TableName=self.jobs_table, Key={"job_id": {"S": job_id}},
                                    ConsistentRead=True).get("Item")
        return _plain(item) if item else None

    def update_job(self, job_id: str, status: str, **values) -> None:
        values = {"status": status, "updated_at": int(time.time()), **values}
        names = {f"#{key}": key for key in values}
        self.client.update_item(
            TableName=self.jobs_table, Key={"job_id": {"S": job_id}},
            UpdateExpression="SET " + ", ".join(f"#{key} = :{key}" for key in values),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues={f":{key}": value for key, value in _item(values).items()},
            ConditionExpression="attribute_exists(job_id)",
        )

    # --- Conversations ---

    @staticmethod
    def _key(user_id: str, conversation_id: str) -> str:
        return f"{user_id}#{conversation_id}"

    def turns(self, user_id: str, conversation_id: str, limit: int = HISTORY_TURNS) -> list[dict]:
        """The latest turns, oldest first."""
        res = self.client.query(
            TableName=self.conversations_table,
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": self._key(user_id, conversation_id)}},
            ScanIndexForward=False, Limit=limit, ConsistentRead=True,
        )
        return [_plain(item) for item in reversed(res.get("Items", []))]

    def history(self, user_id: str, conversation_id: str, limit: int = HISTORY_TURNS) -> list[dict]:
        """The latest turns as chat messages, the form the pipeline takes."""
        messages = []
        for turn in self.turns(user_id, conversation_id, limit):
            messages += [{"role": "user", "content": turn["question"]},
                         {"role": "assistant", "content": turn["answer"]}]
        return messages

    def add_turn(self, user_id: str, conversation_id: str, job_id: str, question: str, answer: str) -> int:
        """Append a turn; a retried job (same job_id) doesn't add it twice."""
        latest = self.turns(user_id, conversation_id, limit=1)
        if latest and latest[-1].get("job_id") == job_id:
            return latest[-1]["turn"]
        turn = latest[-1]["turn"] + 1 if latest else 1
        now = int(time.time())
        self.client.put_item(
            TableName=self.conversations_table,
            Item=_item({"pk": self._key(user_id, conversation_id), "turn": turn, "job_id": job_id,
                        "question": question, "answer": answer, "created_at": now,
                        "expires_at": now + CONVERSATION_TTL_SECONDS}),
            ConditionExpression="attribute_not_exists(pk)",
        )
        return turn
