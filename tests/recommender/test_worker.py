import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QUEUE_URL", "https://sqs.test/queue")

from src.chat import handler as api
from src.chat.store import ChatStore
from src.recommender import worker
from src.recommender.bedrock import ModelError
from src.tracing import capture_spans
from tests.fakes import FakeDynamo, FakeSqs, event, sqs_record


class FakeRecommender:
    def __init__(self, answer="Try Monster.", error=None):
        self.answer_text, self.error, self.calls = answer, error, []

    def answer(self, question, history):
        self.calls.append((question, history))
        if self.error:
            raise self.error
        return self.answer_text


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.store = ChatStore("jobs", "conversations", client=FakeDynamo())
        self.sqs = FakeSqs()
        for module, name in ((api, "_store"), (worker, "_store")):
            patcher = patch.object(module, name, self.store)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(api, "_sqs", self.sqs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, message, conversation_id=None):
        body = {"message": message, **({"conversation_id": conversation_id} if conversation_id else {})}
        return json.loads(api.handler(event(body), None)["body"])

    def run_worker(self, recommender, receive_count=1):
        with patch.object(worker, "recommender", return_value=recommender):
            return worker.handler({"Records": [sqs_record(self.sqs.messages[-1], receive_count)]}, None)

    def test_answers_and_remembers_the_turn(self):
        job = self.ask("Recommend something like Death Note")
        self.assertEqual(self.run_worker(FakeRecommender()), {"batchItemFailures": []})
        stored = self.store.get_job(job["job_id"])
        self.assertEqual((stored["status"], stored["answer"]), ("done", "Try Monster."))
        follow_up = self.ask("Anything shorter?", job["conversation_id"])
        recommender = FakeRecommender("Try Erased.")
        self.run_worker(recommender)
        self.assertEqual(recommender.calls[0], ("Anything shorter?", [
            {"role": "user", "content": "Recommend something like Death Note"},
            {"role": "assistant", "content": "Try Monster."}]))
        self.assertEqual(self.store.get_job(follow_up["job_id"])["answer"], "Try Erased.")

    def test_one_trace_from_api_to_worker(self):
        with capture_spans() as spans:
            job = self.ask("hi")
            self.run_worker(FakeRecommender())
        api_span = next(s for s in spans if s.name == "api.ask")
        job_span = next(s for s in spans if s.name == "recommender.job")
        self.assertEqual(job_span.context.trace_id, api_span.context.trace_id)
        self.assertEqual(job_span.attributes["session.id"], job["conversation_id"])
        self.assertEqual(job_span.attributes["output.value"], "Try Monster.")

    def test_user_facing_errors_fail_the_job_without_retry(self):
        job = self.ask("hi")
        result = self.run_worker(FakeRecommender(error=RuntimeError("No supported recommendations were retrieved.")))
        self.assertEqual(result, {"batchItemFailures": []})
        stored = self.store.get_job(job["job_id"])
        self.assertEqual((stored["status"], stored["error"]), ("failed", "No supported recommendations were retrieved."))

    def test_busy_model_fails_the_job_at_once(self):
        job = self.ask("hi")
        busy = FakeRecommender(error=ModelError(429, "The assistant is busy. Try again in a moment."))
        self.assertEqual(self.run_worker(busy), {"batchItemFailures": []})
        stored = self.store.get_job(job["job_id"])
        self.assertEqual((stored["status"], stored["error"]), ("failed", "The assistant is busy. Try again in a moment."))

    def test_unexpected_errors_are_retried_and_fail_on_the_last_attempt(self):
        job = self.ask("hi")
        broken = FakeRecommender(error=KeyError("bug"))
        self.assertEqual(len(self.run_worker(broken)["batchItemFailures"]), 1)
        self.assertEqual(self.store.get_job(job["job_id"])["status"], "pending")
        self.assertEqual(len(self.run_worker(broken, receive_count=2)["batchItemFailures"]), 1)
        stored = self.store.get_job(job["job_id"])
        self.assertEqual(stored["status"], "failed")
        self.assertNotIn("bug", stored["error"])

    def test_duplicate_delivery_is_ignored(self):
        self.ask("hi")
        recommender = FakeRecommender()
        self.run_worker(recommender)
        self.run_worker(recommender)
        self.assertEqual(len(recommender.calls), 1)
