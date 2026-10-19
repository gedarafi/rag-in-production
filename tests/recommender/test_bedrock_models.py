import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.recommender.providers import BedrockModels


def reply(text):
    return {'choices': [{'message': {'content': text}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


class BedrockModelsTests(unittest.TestCase):
    def setUp(self):
        self.models = BedrockModels('sentence-transformers/all-MiniLM-L6-v2', 'us-east-2', 'qwen.qwen3-32b')
        self.calls = []

    def respond(self, *texts):
        replies = iter(texts)
        def call(body):
            self.calls.append(body)
            return reply(next(replies))
        self.models.chat_completions = call

    def test_schema_steps_send_json_schema_response_format(self):
        self.respond('{"steps": []}')
        schema = {'type': 'object'}
        self.assertEqual(self.models.interpret_request('Recommend comedy', schema), '{"steps": []}')
        body = self.calls[0]
        self.assertEqual(body['model'], 'qwen.qwen3-32b')
        self.assertEqual(body['response_format']['json_schema']['schema'], schema)
        self.assertEqual(body['messages'][-1], {'role': 'user', 'content': 'Recommend comedy'})
        self.assertEqual(body['temperature'], 0)

    def test_invalid_json_is_asked_for_once_more(self):
        self.respond('Sure! here it is', '{"ok": true}')
        self.assertEqual(self.models.interpret_request('q', {'type': 'object'}), '{"ok": true}')
        self.assertEqual(len(self.calls), 2)
        self.assertIn('not valid JSON', self.calls[1]['messages'][-1]['content'])

    def test_plain_answer_has_no_response_format(self):
        self.respond('  Gintama.  ')
        self.assertEqual(self.models.answer('Recommend', 0.2, 100), 'Gintama.')
        self.assertNotIn('response_format', self.calls[0])
        self.assertEqual(self.calls[0]['max_tokens'], 100)

    def test_requests_are_sigv4_signed_for_bedrock_mantle(self):
        credentials = SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(
            access_key='AKID', secret_key='secret', token=None))
        response = MagicMock()
        response.json.return_value = reply('hi')
        with patch('botocore.session.get_session', return_value=SimpleNamespace(get_credentials=lambda: credentials)), \
             patch('src.recommender.providers.requests.post', return_value=response) as post:
            self.assertEqual(self.models.chat_completions({'model': 'm'}), reply('hi'))
        url, kwargs = post.call_args.args[0], post.call_args.kwargs
        self.assertEqual(url, 'https://bedrock-mantle.us-east-2.api.aws/v1/chat/completions')
        self.assertIn('/bedrock-mantle/aws4_request', kwargs['headers']['Authorization'])
        self.assertEqual(json.loads(kwargs['data']), {'model': 'm'})

    def test_pipeline_picks_bedrock_from_the_provider_setting(self):
        from dataclasses import replace
        from src.recommender.config import load_settings
        from src.recommender.rag import RagPipeline
        settings = replace(load_settings(), provider='bedrock', bedrock_model='qwen.qwen3-32b')
        self.assertIsInstance(RagPipeline(settings).models, BedrockModels)

    def test_clone_shares_embedder_but_not_per_request_results(self):
        from dataclasses import replace
        from src.recommender.config import load_settings
        from src.recommender.rag import RagPipeline
        pipeline = RagPipeline(replace(load_settings(), provider='bedrock'))
        pipeline.models.embedder = embedder = object()
        pipeline._chunks = chunks = []
        other = pipeline.clone()
        self.assertIs(other.models.embedder, embedder)
        self.assertIs(other._chunks, chunks)
        other.models.last_fact_grounding = {'a': 1}
        self.assertFalse(hasattr(pipeline.models, 'last_fact_grounding'))


if __name__ == '__main__':
    unittest.main()
