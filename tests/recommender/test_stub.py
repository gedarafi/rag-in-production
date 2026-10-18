import json
import unittest

from src.recommender.bedrock import BedrockChat
from src.recommender.stub import STUB_PLAN, StubTransport, minimal, titles
from src.recommender.tool_plan import plan_schema, validate_plan


class StubTests(unittest.TestCase):
    def test_minimal_values_satisfy_their_schema(self):
        import jsonschema
        schema = {"type": "object", "required": ["kind", "facts", "answer", "n"], "properties": {
            "kind": {"const": "prose"}, "facts": {"type": "array", "minItems": 1, "items": {"enum": ["a", "b"]}},
            "answer": {"type": "string", "minLength": 3, "pattern": r".*\?.*"}, "n": {"type": "integer", "minimum": 2}}}
        value = minimal(schema)
        jsonschema.validate(value, schema)
        self.assertEqual(value["facts"], ["a"])

    def test_the_plan_is_a_valid_recommendation_plan(self):
        schema = plan_schema(8)
        self.assertEqual(validate_plan(json.dumps(STUB_PLAN), schema)["steps"][0]["tool"], "recommend_anime")
        chat = BedrockChat("stub", StubTransport())
        self.assertEqual(json.loads(chat.json("plan_tools", "Plan.", [{"role": "user", "content": "q"}], schema)), STUB_PLAN)

    def test_text_lists_the_titles_in_the_evidence(self):
        chat = BedrockChat("stub", StubTransport())
        reply = chat.text("agent_reply", "s", [{"role": "user", "content": 'x {"title": "Orbitfall"} **Clockwork Lighthouse**'}],
                          temperature=0, max_tokens=10)
        self.assertEqual(reply, "From the catalog: Orbitfall, Clockwork Lighthouse.")
        self.assertEqual(titles("nothing"), [])
