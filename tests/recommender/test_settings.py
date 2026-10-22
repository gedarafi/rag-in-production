import os
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.recommender.config import load_settings

VARIABLES = ('RAG_PROVIDER', 'RAG_LLM_MODEL', 'RAG_EMBEDDING_MODEL', 'RAG_OLLAMA_URL', 'RAG_PHOENIX_ENDPOINT',
             'RAG_PHOENIX_PROJECT', 'RAG_BEDROCK_REGION', 'AWS_REGION', 'RAG_BEDROCK_MODEL')


class SettingsTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in VARIABLES:
            os.environ.pop(name, None)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.config = Path(self.folder.name) / 'settings.toml'
        self.text = Path('settings.toml').read_text(encoding='utf-8')

    def load(self, text=None):
        self.config.write_text(self.text if text is None else text, encoding='utf-8')
        return load_settings(self.config, 'anilist')

    def test_environment_overrides_the_file(self):
        os.environ.update(RAG_PROVIDER='bedrock', AWS_REGION='eu-west-1', RAG_OLLAMA_URL='http://ollama:11434/')
        settings = self.load()
        self.assertEqual((settings.provider, settings.bedrock_region), ('bedrock', 'eu-west-1'))
        self.assertEqual(settings.ollama_url, 'http://ollama:11434')
        os.environ['RAG_BEDROCK_REGION'] = 'us-west-2'
        self.assertEqual(self.load().bedrock_region, 'us-west-2')

    def test_an_empty_variable_still_overrides(self):
        os.environ['RAG_PHOENIX_ENDPOINT'] = ''
        self.assertEqual(self.load(self.text.replace('endpoint = ""', 'endpoint = "http://x/v1/traces"')).tracing_endpoint, '')

    def test_invalid_values_name_the_setting(self):
        for old, new, field in (('router_top_k = 6', 'router_top_k = 1', 'router_top_k'),
                                ('provider = "ollama"', 'provider = "openai"', 'provider'),
                                ('embedding_device = "auto"', 'embedding_device = "tpu"', 'embedding_device'),
                                ('max_attempts = 4', 'max_attempts = 0', 'llm_max_attempts')):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.load(self.text.replace(old, new))
        os.environ['RAG_PROVIDER'] = 'openai'
        with self.assertRaisesRegex(ValueError, 'provider'):
            self.load()

    def test_replace_validates_too(self):
        with self.assertRaisesRegex(ValueError, 'context_chars'):
            replace(self.load(), context_chars=10)

    def test_older_settings_files_get_defaults(self):
        without = re.sub(r'(?ms)^\[llm\].*?(?=^\[)', '', self.text)
        settings = self.load(without)
        self.assertEqual((settings.llm_max_attempts, settings.llm_connect_timeout, settings.prompt_overrides),
                         (4, 5.0, None))

    def test_security_llm_review_is_on_unless_turned_off(self):
        self.assertTrue(self.load(re.sub(r'(?ms)^\[security\].*?(?=^\[)', '', self.text)).security_llm_review)
        self.assertFalse(self.load(self.text.replace('llm_review = true', 'llm_review = false')).security_llm_review)

    def test_prompt_overrides_resolve_against_the_settings_folder(self):
        settings = self.load(self.text.replace('# prompt_overrides = ""', 'prompt_overrides = "prompts/mine"'))
        self.assertEqual(settings.prompt_overrides, (self.config.parent / 'prompts/mine').resolve())


if __name__ == '__main__':
    unittest.main()
