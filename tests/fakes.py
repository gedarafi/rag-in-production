"""Stand-ins for Bedrock, DynamoDB, SQS and API Gateway, shared by the unit tests and the Phoenix test."""

import base64
import io
import json
import urllib.error


class AwsError(Exception):
    """Shaped like botocore's ClientError, so the tests need no boto3."""

    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class FakeBedrock:
    def __init__(self, reply="Week 3 covers recursion.", error=None, usage=None):
        self.reply, self.error, self.usage, self.calls = reply, error, usage, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        res = {"output": {"message": {"role": "assistant", "content": [{"text": self.reply}]}}}
        if self.usage:
            res["usage"] = self.usage
        return res


class FakeMantle:
    def __init__(self, reply="Week 3 covers recursion.", error=None, usage=None):
        self.reply, self.error, self.usage, self.calls = reply, error, usage, []

    def chat_completions(self, **body):
        self.calls.append(body)
        if self.error:
            raise self.error
        res = {"choices": [{"index": 0, "message": {"role": "assistant", "content": self.reply}}]}
        if self.usage:
            res["usage"] = self.usage
        return res


def http_error(status) -> urllib.error.HTTPError:
    body = io.BytesIO(json.dumps({"error": {"code": str(status), "message": "nope"}}).encode())
    return urllib.error.HTTPError("https://bedrock-mantle.example/v1/chat/completions", status, "error", {}, body)


class ConditionFailed(AwsError):
    def __init__(self):
        super().__init__("ConditionalCheckFailedException")


class FakeDynamo:
    """The DynamoDB calls of src/chat/store.py, kept in memory: jobs by job_id, turns by (pk, turn)."""

    KEYS = {"jobs": ("job_id",), "conversations": ("pk", "turn")}

    def __init__(self):
        self.tables = {"jobs": {}, "conversations": {}}

    def _key(self, table, item):
        return tuple(next(iter(item[name].values())) for name in self.KEYS[table])

    def put_item(self, TableName, Item, ConditionExpression=None):
        key = self._key(TableName, Item)
        if ConditionExpression and ConditionExpression.startswith("attribute_not_exists") and key in self.tables[TableName]:
            raise ConditionFailed()
        self.tables[TableName][key] = dict(Item)

    def get_item(self, TableName, Key, ConsistentRead=False):
        item = self.tables[TableName].get(self._key(TableName, Key))
        return {"Item": dict(item)} if item else {}

    def update_item(self, TableName, Key, UpdateExpression, ExpressionAttributeNames, ExpressionAttributeValues,
                    ConditionExpression=None):
        key = self._key(TableName, Key)
        if key not in self.tables[TableName]:
            raise ConditionFailed()
        for assignment in UpdateExpression.removeprefix("SET ").split(", "):
            name, value = assignment.split(" = ")
            self.tables[TableName][key][ExpressionAttributeNames[name]] = ExpressionAttributeValues[value]

    def query(self, TableName, KeyConditionExpression, ExpressionAttributeValues, ScanIndexForward=True,
              Limit=None, ConsistentRead=False):
        pk = ExpressionAttributeValues[":pk"]["S"]
        items = sorted((item for (p, _), item in self.tables[TableName].items() if p == pk),
                       key=lambda item: int(item["turn"]["N"]), reverse=not ScanIndexForward)
        return {"Items": items[:Limit] if Limit else items}


class FakeSqs:
    def __init__(self):
        self.messages = []

    def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return {"MessageId": f"m{len(self.messages)}"}


def sqs_record(message: dict, receive_count=1) -> dict:
    """The SQS event record the worker receives for a message the api Lambda sent."""
    return {"messageId": "m1", "body": message["MessageBody"],
            "attributes": {"ApproximateReceiveCount": str(receive_count)},
            "messageAttributes": {key: {"stringValue": value["StringValue"], "dataType": "String"}
                                  for key, value in message.get("MessageAttributes", {}).items()}}


def event(body, sub="user-123", base64_encoded=False) -> dict:
    claims = {"sub": sub, "client_id": "client"} if sub else {}
    raw = body if isinstance(body, str) else json.dumps(body)
    if base64_encoded:
        raw = base64.b64encode(raw.encode()).decode()
    return {
        "body": raw,
        "isBase64Encoded": base64_encoded,
        "requestContext": {"http": {"method": "POST"}, "authorizer": {"jwt": {"claims": claims}}},
    }


def get_event(job_id, sub="user-123") -> dict:
    claims = {"sub": sub, "client_id": "client"} if sub else {}
    return {
        "pathParameters": {"job_id": job_id},
        "requestContext": {"http": {"method": "GET"}, "authorizer": {"jwt": {"claims": claims}}},
    }
