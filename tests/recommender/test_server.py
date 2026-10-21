import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from src.recommender import server
from src.recommender.errors import UserFacingError

ISSUER = 'https://cognito-idp.us-east-2.amazonaws.com/us-east-2_pool'
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def token(**claims):
    body = {'iss': ISSUER, 'sub': 'user-1', 'token_use': 'access', 'client_id': 'web-client',
            'exp': int(time.time()) + 600, **claims}
    return jwt.encode(body, KEY, algorithm='RS256', headers={'kid': 'k1'})


class Pipeline:
    def __init__(self, answer=None, error=None, gate=None):
        self.answer, self.error, self.gate, self.asked = answer, error, gate, []

    def ask(self, question, history):
        self.asked.append((question, history))
        if self.gate:
            self.gate.wait(5)
        if self.error:
            raise self.error
        return self.answer


class Pool:
    def __init__(self, pipeline):
        self.pipeline, self.busy = pipeline, False

    def acquire(self, timeout):
        if self.busy:
            return None
        self.busy = True
        return self.pipeline

    def release(self, pipeline):
        self.busy = False


class ServerTests(unittest.TestCase):
    def client(self, pipeline):
        verifier = server.TokenVerifier(ISSUER, 'web-client', 'https://keys.invalid/jwks.json')
        verifier.jwks = SimpleNamespace(get_signing_key_from_jwt=lambda _: SimpleNamespace(key=KEY.public_key()))
        self.pool = Pool(pipeline)
        app = server.create_app(verifier, lambda: (self.pool, 'v1'))
        client = TestClient(app)
        for _ in range(100):  # the pool loads in a background thread
            if client.get('/recommend/health').status_code == 200:
                break
            time.sleep(0.01)
        return client

    def ask(self, client, authorization, **body):
        return client.post('/recommend/chat', json={'message': 'Recommend a comedy', **body},
                           headers={'Authorization': authorization} if authorization else {})

    def test_health_needs_no_token(self):
        response = self.client(Pipeline('x')).get('/recommend/health')
        self.assertEqual(response.json(), {'ready': True, 'version': 'v1', 'error': ''})

    def test_signed_in_user_gets_the_answer_as_the_last_line(self):
        pipeline = Pipeline('Try Gintama.')
        history = [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]
        response = self.ask(self.client(pipeline), 'Bearer ' + token(), history=history)
        self.assertEqual(response.status_code, 200)
        last = json.loads(response.text.strip().splitlines()[-1])
        self.assertEqual(last, {'type': 'answer', 'answer': 'Try Gintama.'})
        self.assertEqual(pipeline.asked, [('Recommend a comedy', history)])
        self.assertFalse(self.pool.busy)

    def test_bad_tokens_are_rejected(self):
        client = self.client(Pipeline('x'))
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = jwt.encode({'iss': ISSUER, 'sub': 'u', 'token_use': 'access', 'client_id': 'web-client',
                             'exp': int(time.time()) + 600}, other, algorithm='RS256')
        for authorization in (None, 'Basic abc', 'Bearer ' + token(exp=int(time.time()) - 10),
                              'Bearer ' + token(iss='https://evil.example'), 'Bearer ' + token(token_use='id'),
                              'Bearer ' + token(client_id='other-client'), 'Bearer ' + forged):
            with self.subTest(authorization=authorization):
                response = self.ask(client, authorization)
                self.assertEqual(response.status_code, 401)
                self.assertIn('Sign in again', response.json()['error'])

    def test_full_pool_answers_busy(self):
        gate = threading.Event()
        client = self.client(Pipeline('x', gate=gate))
        self.pool.busy = True
        response = self.ask(client, 'Bearer ' + token())
        self.assertEqual(response.status_code, 429)
        gate.set()

    def test_throttled_model_reports_busy_and_frees_the_pipeline(self):
        throttled = requests.HTTPError(response=SimpleNamespace(status_code=429))
        failure = RuntimeError('Could not reach the model')
        failure.__cause__ = throttled
        client = self.client(Pipeline(error=failure))
        with patch('builtins.print'):
            response = self.ask(client, 'Bearer ' + token())
        last = json.loads(response.text.strip().splitlines()[-1])
        self.assertEqual((last['type'], last['status']), ('error', 429))
        self.assertFalse(self.pool.busy)

    def last_error(self, failure):
        client = self.client(Pipeline(error=failure))
        with patch('builtins.print'):
            response = self.ask(client, 'Bearer ' + token())
        return json.loads(response.text.strip().splitlines()[-1])

    def test_user_facing_failures_reach_the_page(self):
        message = 'No supported recommendations were retrieved. Please rephrase your preferences and retry.'
        wrapped = RuntimeError('Agent failed')
        wrapped.__cause__ = UserFacingError(message)
        for failure in (UserFacingError(message), wrapped):
            with self.subTest(failure=failure):
                self.assertEqual(self.last_error(failure), {'type': 'error', 'status': 502, 'error': message})

    def test_internal_failures_stay_generic(self):
        last = self.last_error(RuntimeError('Could not reach Ollama at http://10.0.0.5:11434: refused'))
        self.assertEqual(last['error'], "The assistant couldn't answer. Try again.")

    def test_validation_rejects_oversized_messages(self):
        response = self.ask(self.client(Pipeline('x')), 'Bearer ' + token(), message='x' * 2001)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
