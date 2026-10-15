"""Placeholder chat endpoint: POST /api/chat with {"message": "..."}.

API Gateway has already validated the Cognito token before this runs. Replace answer()
with the RAG pipeline (retrieve from the knowledge base, then generate) later.
"""

import base64
import json

MAX_MESSAGE_CHARS = 4000

UNDER_CONSTRUCTION = (
    "The assistant is under construction. Your sign-in works, and soon I will answer "
    "questions from the course documents."
)


def answer(message: str, user_id: str) -> str:
    return UNDER_CONSTRUCTION


def response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def handler(event, context):
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    user_id = claims.get("sub")
    if not user_id:
        return response(401, {"error": "Not signed in."})

    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return response(400, {"error": "Body must be JSON."})

    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        return response(400, {"error": "Send a non-empty 'message'."})
    if len(message) > MAX_MESSAGE_CHARS:
        return response(400, {"error": f"Messages are limited to {MAX_MESSAGE_CHARS} characters."})

    return response(200, {"answer": answer(message.strip(), user_id), "status": "under_construction"})
