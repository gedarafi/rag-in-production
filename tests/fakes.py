"""Stand-ins for Bedrock and API Gateway, shared by the unit tests and the Phoenix test."""

import base64
import io
import json
import urllib.error


class AwsError(Exception):
    """Shaped like botocore's ClientError, so the tests need no boto3."""

    def __init__(self, code, message=None):
        super().__init__(message or code)
        self.response = {"Error": {"Code": code, "Message": message or code}}


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


def http_error(status, message="nope") -> urllib.error.HTTPError:
    body = io.BytesIO(json.dumps({"error": {"code": str(status), "message": message}}).encode())
    return urllib.error.HTTPError("https://bedrock-mantle.example/v1/chat/completions", status, "error", {}, body)


def event(body, sub="user-123", base64_encoded=False) -> dict:
    claims = {"sub": sub, "client_id": "client"} if sub else {}
    raw = body if isinstance(body, str) else json.dumps(body)
    if base64_encoded:
        raw = base64.b64encode(raw.encode()).decode()
    return {
        "body": raw,
        "isBase64Encoded": base64_encoded,
        "requestContext": {"authorizer": {"jwt": {"claims": claims}}},
    }
