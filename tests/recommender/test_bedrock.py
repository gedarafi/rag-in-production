import json
import unittest
import urllib.error

from src.recommender.bedrock import (BedrockChat, Converse, InvalidOutput, Mantle, ModelError, extract_json)
from src.tracing import capture_spans
from tests.fakes import AwsError, FakeBedrock, http_error
from tests.recommender.bedrock_stub import ScriptedTransport

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["action"],
          "properties": {"action": {"enum": ["recommend", "lookup"]}}}


class JsonTests(unittest.TestCase):
    def test_extracts_json_from_fences_and_prose(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Here you go: {"a": [1, 2]} hope it helps'), {"a": [1, 2]})
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_valid_reply_needs_one_call(self):
        transport = ScriptedTransport('{"action": "lookup"}')
        self.assertEqual(json.loads(BedrockChat("m", transport).json("plan", "Plan.", [{"role": "user", "content": "q"}],
                                                                     SCHEMA)), {"action": "lookup"})
        self.assertEqual(len(transport.calls), 1)
        self.assertIn('"enum":["recommend","lookup"]', transport.calls[0]["system"])
        self.assertEqual(transport.calls[0]["schema"], SCHEMA)

    def test_invalid_reply_is_corrected_once(self):
        transport = ScriptedTransport('{"action": "watch"}', '{"action": "recommend"}')
        with capture_spans() as spans:
            reply = BedrockChat("m", transport).json("plan", "Plan.", [{"role": "user", "content": "q"}], SCHEMA)
        self.assertEqual(json.loads(reply), {"action": "recommend"})
        correction = transport.calls[1]["messages"]
        self.assertEqual(correction[-2], {"role": "assistant", "content": '{"action": "watch"}'})
        self.assertIn("invalid", correction[-1]["content"])
        self.assertEqual([s.name for s in spans], ["plan", "plan.correction"])

    def test_still_invalid_raises_a_value_error(self):
        with self.assertRaises(InvalidOutput) as raised:
            BedrockChat("m", ScriptedTransport("nope")).json("plan", "Plan.", [{"role": "user", "content": "q"}], SCHEMA)
        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(raised.exception.status, 502)


class SpanTests(unittest.TestCase):
    def test_llm_span_has_the_chat_and_token_counts(self):
        with capture_spans() as spans:
            BedrockChat("test-model", ScriptedTransport("An answer")).text(
                "agent_reply", "Be brief.", [{"role": "user", "content": "Who?"}], temperature=0.2, max_tokens=100)
        (span,) = spans
        a = span.attributes
        self.assertEqual(span.name, "agent_reply")
        self.assertEqual(a["openinference.span.kind"], "LLM")
        self.assertEqual(a["llm.model_name"], "test-model")
        self.assertEqual(a["llm.input_messages.0.message.content"], "Be brief.")
        self.assertEqual(a["llm.input_messages.1.message.content"], "Who?")
        self.assertEqual(a["llm.output_messages.0.message.content"], "An answer")
        self.assertEqual([a[f"llm.token_count.{k}"] for k in ("prompt", "completion", "total")], [10, 5, 15])

    def test_failed_call_is_an_error_span_and_a_model_error(self):
        with capture_spans() as spans, self.assertRaises(ModelError) as raised:
            BedrockChat("m", ScriptedTransport(error=AwsError("ThrottlingException"))).text(
                "answer", "s", [{"role": "user", "content": "q"}], temperature=0, max_tokens=10)
        self.assertEqual(raised.exception.status, 429)
        self.assertEqual(spans[0].status.status_code.name, "ERROR")

    def test_empty_reply_is_an_error(self):
        with self.assertRaises(ModelError):
            BedrockChat("m", ScriptedTransport("  ")).text("answer", "s", [{"role": "user", "content": "q"}],
                                                          temperature=0, max_tokens=10)


class ErrorTests(unittest.TestCase):
    def call(self, error):
        with self.assertRaises(ModelError) as raised:
            BedrockChat("m", ScriptedTransport(error=error)).text("a", "s", [{"role": "user", "content": "q"}],
                                                                 temperature=0, max_tokens=1)
        return raised.exception.status

    def test_statuses(self):
        self.assertEqual(self.call(AwsError("ServiceUnavailableException")), 429)
        self.assertEqual(self.call(AwsError("ValidationException")), 502)
        self.assertEqual(self.call(TimeoutError("read timeout")), 504)
        self.assertEqual(self.call(http_error(429)), 429)
        self.assertEqual(self.call(http_error(400)), 502)


class TransportTests(unittest.TestCase):
    def test_converse_request_and_usage(self):
        fake = FakeBedrock(reply="Hi", usage={"inputTokens": 3, "outputTokens": 1, "totalTokens": 4})
        reply = Converse(client=fake).complete("model-x", "System.", [{"role": "user", "content": "q"}], 50, 0.1)
        self.assertEqual((reply.text, reply.total_tokens), ("Hi", 4))
        call = fake.calls[0]
        self.assertEqual(call["modelId"], "model-x")
        self.assertEqual(call["system"], [{"text": "System."}])
        self.assertEqual(call["messages"], [{"role": "user", "content": [{"text": "q"}]}])
        self.assertEqual(call["inferenceConfig"], {"maxTokens": 50, "temperature": 0.1})

    def mantle(self, *results):
        mantle = Mantle.__new__(Mantle)
        mantle.loose, mantle.bodies, results = set(), [], list(results)

        def post(body):
            mantle.bodies.append(json.loads(json.dumps(body)))
            result = results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        mantle.post = post
        return mantle

    def test_mantle_asks_for_strict_json_schema(self):
        mantle = self.mantle({"choices": [{"message": {"content": '{"action": "lookup"}'}}],
                              "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})
        reply = mantle.complete("qwen", "Plan.", [{"role": "user", "content": "q"}], 100, 0, SCHEMA)
        self.assertEqual(reply.total_tokens, 5)
        body = mantle.bodies[0]
        self.assertEqual(body["messages"][0], {"role": "system", "content": "Plan."})
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SCHEMA)

    def test_mantle_falls_back_to_loose_schema_once_refused(self):
        ok = {"choices": [{"message": {"content": "{}"}}]}
        mantle = self.mantle(http_error(400), ok, ok)
        mantle.complete("qwen", "s", [{"role": "user", "content": "q"}], 10, 0, SCHEMA)
        mantle.complete("qwen", "s", [{"role": "user", "content": "q"}], 10, 0, SCHEMA)
        self.assertEqual([b["response_format"]["json_schema"]["strict"] for b in mantle.bodies], [True, False, False])

    def test_mantle_null_content_is_empty(self):
        mantle = self.mantle({"choices": [{"message": {"content": None}}]})
        self.assertEqual(mantle.complete("m", "s", [{"role": "user", "content": "q"}], 10, 0).text, "")

    def test_mantle_other_errors_propagate(self):
        mantle = self.mantle(http_error(503))
        with self.assertRaises(urllib.error.HTTPError):
            mantle.complete("m", "s", [{"role": "user", "content": "q"}], 10, 0, SCHEMA)


class OllamaTests(unittest.TestCase):
    def test_constrains_output_to_the_schema(self):
        import io
        from unittest.mock import patch
        from src.recommender.bedrock import Ollama
        reply = {"message": {"content": '{"action": "lookup"}'}, "prompt_eval_count": 7, "eval_count": 3}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            result = Ollama("http://ollama:11434/").complete("llama3.2:3b", "Plan.", [{"role": "user", "content": "q"}],
                                                             100, 0, SCHEMA)
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://ollama:11434/api/chat")
        self.assertEqual(body["format"], SCHEMA)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "Plan."})
        self.assertEqual(body["options"]["num_predict"], 100)
        self.assertEqual((result.text, result.total_tokens), ('{"action": "lookup"}', 10))

    def test_unreachable_ollama_is_a_timeout_error(self):
        import urllib.error
        from unittest.mock import patch
        from src.recommender.bedrock import Ollama
        chat = BedrockChat("llama3.2:3b", Ollama("http://ollama:11434"))
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
                self.assertRaises(ModelError) as raised:
            chat.text("answer", "s", [{"role": "user", "content": "q"}], temperature=0, max_tokens=5)
        self.assertEqual(raised.exception.status, 504)
