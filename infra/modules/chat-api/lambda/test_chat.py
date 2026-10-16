import base64
import json
import os
import unittest

import chat
from chat import MAX_MESSAGE_CHARS, SYSTEM_PROMPT, handler


class AwsError(Exception):
    """Shaped like botocore's ClientError, so the tests need no boto3."""

    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class FakeBedrock:
    def __init__(self, reply="Week 3 covers recursion.", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"output": {"message": {"role": "assistant", "content": [{"text": self.reply}]}}}


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


def call(evt) -> tuple[int, dict]:
    result = handler(evt, None)
    return result["statusCode"], json.loads(result["body"])


class ChatTest(unittest.TestCase):
    def setUp(self):
        self.bedrock = FakeBedrock()
        chat._client = self.bedrock
        os.environ["MODEL_ID"] = "test-model"
        self.addCleanup(setattr, chat, "_client", None)
        self.addCleanup(os.environ.pop, "MODEL_ID", None)

    def test_answers_with_the_model(self):
        status, body = call(event({"message": "  What is in week 3?  "}))
        self.assertEqual(status, 200)
        self.assertEqual(body, {"answer": "Week 3 covers recursion."})

        (request,) = self.bedrock.calls
        self.assertEqual(request["modelId"], "test-model")
        self.assertEqual(request["system"], [{"text": SYSTEM_PROMPT}])
        self.assertEqual(request["messages"], [{"role": "user", "content": [{"text": "What is in week 3?"}]}])
        self.assertEqual(request["inferenceConfig"]["maxTokens"], 512)

    def test_max_tokens_is_configurable(self):
        os.environ["MAX_TOKENS"] = "64"
        self.addCleanup(os.environ.pop, "MAX_TOKENS", None)
        call(event({"message": "hi"}))
        self.assertEqual(self.bedrock.calls[0]["inferenceConfig"]["maxTokens"], 64)

    def test_accepts_base64_body(self):
        status, _ = call(event({"message": "hi"}, base64_encoded=True))
        self.assertEqual(status, 200)

    def test_rejects_missing_claims(self):
        status, _ = call(event({"message": "hi"}, sub=None))
        self.assertEqual(status, 401)
        self.assertEqual(self.bedrock.calls, [])

    def test_rejects_bad_input(self):
        for body in ["not json", {"message": ""}, {"message": "   "}, {"message": 5}, {}, [1, 2]]:
            with self.subTest(body=body):
                status, payload = call(event(body))
                self.assertEqual(status, 400)
                self.assertIn("error", payload)
        self.assertEqual(self.bedrock.calls, [])

    def test_rejects_too_long_message(self):
        status, _ = call(event({"message": "x" * (MAX_MESSAGE_CHARS + 1)}))
        self.assertEqual(status, 400)

    def test_model_errors(self):
        cases = [
            (AwsError("ThrottlingException"), 429),
            (AwsError("ValidationException"), 502),
            (AwsError("AccessDeniedException"), 502),
            (TimeoutError("read timeout"), 504),
        ]
        for error, expected in cases:
            with self.subTest(error=error):
                chat._client = FakeBedrock(error=error)
                status, body = call(event({"message": "hi"}))
                self.assertEqual(status, expected)
                self.assertIn("error", body)
                self.assertNotIn("Exception", body["error"])  # no AWS internals in the reply

    def test_empty_model_answer_is_an_error(self):
        chat._client = FakeBedrock(reply="   ")
        status, _ = call(event({"message": "hi"}))
        self.assertEqual(status, 502)

    def test_never_cached(self):
        result = handler(event({"message": "hi"}), None)
        self.assertEqual(result["headers"]["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
