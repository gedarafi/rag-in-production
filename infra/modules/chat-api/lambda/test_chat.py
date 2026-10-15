import base64
import json
import unittest

from chat import MAX_MESSAGE_CHARS, UNDER_CONSTRUCTION, handler


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
    def test_answers_under_construction(self):
        status, body = call(event({"message": "What is in week 3?"}))
        self.assertEqual(status, 200)
        self.assertEqual(body["answer"], UNDER_CONSTRUCTION)
        self.assertEqual(body["status"], "under_construction")

    def test_accepts_base64_body(self):
        status, _ = call(event({"message": "hi"}, base64_encoded=True))
        self.assertEqual(status, 200)

    def test_rejects_missing_claims(self):
        status, _ = call(event({"message": "hi"}, sub=None))
        self.assertEqual(status, 401)

    def test_rejects_bad_input(self):
        for body in ["not json", {"message": ""}, {"message": "   "}, {"message": 5}, {}, [1, 2]]:
            with self.subTest(body=body):
                status, payload = call(event(body))
                self.assertEqual(status, 400)
                self.assertIn("error", payload)

    def test_rejects_too_long_message(self):
        status, _ = call(event({"message": "x" * (MAX_MESSAGE_CHARS + 1)}))
        self.assertEqual(status, 400)

    def test_never_cached(self):
        result = handler(event({"message": "hi"}), None)
        self.assertEqual(result["headers"]["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
