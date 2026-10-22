import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from src.recommender.providers import LocalModels, RetryPolicy, retry_after, transient
from src.recommender.server import http_status


def response(status, body=None, headers=None):
    reply = requests.Response()
    reply.status_code = status
    reply.headers.update(headers or {})
    reply._content = b'{"response": "Gintama."}' if body is None else body
    return reply


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.models = LocalModels('unused', 'http://ollama.test', 'test',
                                  retry=RetryPolicy(max_attempts=3, backoff_seconds=1, max_backoff_seconds=8))
        self.waits = []
        self.models.sleep = self.waits.append

    def answer(self, *replies):
        with patch('src.recommender.providers.requests.post', side_effect=list(replies)) as post, \
                patch('src.recommender.providers.event') as event:
            try:
                return self.models.answer('Recommend', 0.2, 50)
            finally:
                self.posts, self.events = post.call_args_list, event.call_args_list

    def test_throttling_is_retried_after_a_backoff(self):
        self.assertEqual(self.answer(response(503), response(200)), 'Gintama.')
        self.assertEqual(len(self.posts), 2)
        self.assertEqual(len(self.waits), 1)
        self.assertTrue(0 <= self.waits[0] <= 1)
        self.assertEqual(self.events[0].args[0], 'llm_retry')
        self.assertEqual(self.events[0].kwargs['attempt'], 1)

    def test_retry_after_is_honoured_up_to_the_maximum_backoff(self):
        self.answer(response(429, headers={'Retry-After': '3'}), response(429, headers={'Retry-After': '60'}),
                    response(200))
        self.assertEqual(self.waits, [3.0, 8])

    def test_dropped_connections_are_retried(self):
        self.assertEqual(self.answer(requests.ConnectionError('reset'), response(200)), 'Gintama.')

    def test_client_errors_fail_at_once(self):
        with self.assertRaises(RuntimeError) as raised:
            self.answer(response(400), response(200))
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.waits, [])
        self.assertEqual(raised.exception.__cause__.response.status_code, 400)

    def test_the_last_error_survives_for_the_web_service(self):
        with self.assertRaises(RuntimeError) as raised:
            self.answer(response(429), response(429), response(429))
        self.assertEqual(len(self.posts), 3)
        self.assertEqual(http_status(raised.exception), 429)

    def test_each_attempt_has_a_connect_and_a_read_timeout(self):
        self.answer(response(200))
        self.assertEqual(self.posts[0].kwargs['timeout'], (5.0, 300))

    def test_helpers(self):
        self.assertTrue(transient(requests.Timeout()))
        self.assertTrue(transient(requests.HTTPError(response=SimpleNamespace(status_code=502))))
        self.assertFalse(transient(requests.HTTPError(response=SimpleNamespace(status_code=403))))
        self.assertFalse(transient(ValueError('bad JSON')))
        self.assertEqual(retry_after(requests.HTTPError(response=response(429, headers={'Retry-After': '2'}))), 2.0)
        self.assertIsNone(retry_after(requests.HTTPError(response=response(429, headers={'Retry-After': 'soon'}))))

    def test_policy_comes_from_the_settings(self):
        from src.recommender.config import load_settings
        from src.recommender.rag import RagPipeline
        self.assertEqual(RagPipeline(load_settings()).models.retry, RetryPolicy(4, 1.0, 20.0, 5.0))


if __name__ == '__main__':
    unittest.main()
