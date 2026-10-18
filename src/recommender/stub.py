"""A deterministic stand-in for the Bedrock model (MODEL_API=stub), for LocalStack e2e runs and CI.

It answers in milliseconds, so a test exercises the whole path (API, queue, worker, bundle, pipeline,
DynamoDB, traces) without a CPU-bound model. Its answers say nothing about quality: those are checked
with a real model (`recommender ask`, the evaluations, or the deployed stack on AWS).

Structured steps get the smallest reply valid for their JSON schema, except the tool plan, which asks
the catalog for recommendations; text steps list the titles found in the evidence they are given.
"""
from __future__ import annotations

import json
import re

from .bedrock import Reply

STUB_PLAN = {"scope": "in_domain", "direct_response": "", "use_previous_context": False,
             "goal": "Recommend anime from the catalog", "clarification": "",
             "steps": [{"tool": "recommend_anime", "arguments": {}}]}


def minimal(schema: dict):
    """The smallest value that satisfies a (simple) JSON schema."""
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return minimal(schema["anyOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = kind[0]
    if kind == "object" or "properties" in schema:
        return {key: minimal(schema["properties"][key]) for key in schema.get("required", [])}
    if kind == "array":
        return [minimal(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
    if kind in ("integer", "number"):
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    text = "?" if "?" in schema.get("pattern", "") else "stub"
    return (text * max(1, schema.get("minLength", 1)))[:schema.get("maxLength", 1000)]


def titles(text: str) -> list[str]:
    """Anime titles in the evidence: JSON "title" fields and **bold** names."""
    found = re.findall(r'"title":\s*"([^"]+)"', text) + re.findall(r"\*\*([^*\n]+)\*\*", text)
    return list(dict.fromkeys(found))


class StubTransport:
    system = "stub"

    def complete(self, model, system, messages, max_tokens, temperature, schema=None, name=None) -> Reply:
        prompt = "\n".join(m["content"] for m in messages)
        if schema is not None:
            value = STUB_PLAN if (name or "").startswith("plan_tools") else minimal(schema)
            text = json.dumps(value)
        else:
            found = titles(prompt)[:5]
            text = ("From the catalog: " + ", ".join(found) + "." if found
                    else "I couldn't find that in the catalog. Which anime do you mean?")
        return Reply(text, len(prompt) // 4, len(text) // 4, (len(prompt) + len(text)) // 4)
