import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from src.cli import main


class CliDebugTests(unittest.TestCase):
    def run_cli(self, debug=False, fail=False):
        route = {'llm_calls': 2, 'use_previous_context': False,
                 'router_plan': {'steps': []}, 'agent_trace': []}
        def ask(question):
            if fail:
                raise RuntimeError('Model unavailable')
            return 'A natural answer.'
        pipeline = SimpleNamespace(ask=ask, last_route=route)
        argv = ['rag', 'ask', 'Who is Naruto?', '--kb', 'anilist'] + (['--debug'] if debug else [])
        output, errors = io.StringIO(), io.StringIO()
        with patch('sys.argv', argv), patch('src.rag.RagPipeline', return_value=pipeline), redirect_stdout(output), redirect_stderr(errors):
            if fail:
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 1)
            else:
                main()
        return output.getvalue(), errors.getvalue()

    def test_debug_is_disabled_by_default(self):
        output, errors = self.run_cli()
        self.assertEqual(output, 'A natural answer.\n')
        self.assertEqual(errors, '')

    def test_debug_goes_to_stderr_and_preserves_stdout(self):
        output, errors = self.run_cli(debug=True)
        self.assertEqual(output, 'A natural answer.\n')
        data = json.loads(errors)
        self.assertEqual(data['routing']['llm_calls'], 2)
        self.assertFalse(data['routing']['use_previous_context'])
        self.assertIn('elapsed_seconds', data)

    def test_failed_queries_keep_available_routing_diagnostics(self):
        output, errors = self.run_cli(debug=True, fail=True)
        self.assertEqual(output, '')
        data, offset = json.JSONDecoder().raw_decode(errors)
        self.assertEqual(data['routing']['llm_calls'], 2)
        self.assertIn('Error: Model unavailable', errors[offset:])
