import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from src.recommender.cli import main
from src.tracing import capture_spans


class CliTests(unittest.TestCase):
    def run_cli(self, fail=False, exporting=False):
        def ask(question):
            if fail:
                raise RuntimeError('Model unavailable')
            return 'A natural answer.'
        pipeline = SimpleNamespace(ask=ask, last_route={})
        output, errors = io.StringIO(), io.StringIO()
        with patch('src.recommender.rag.RagPipeline', return_value=pipeline), \
             patch('src.tracing.setup_tracing', return_value=exporting), \
             capture_spans(), redirect_stdout(output), redirect_stderr(errors):
            if fail:
                with self.assertRaises(SystemExit) as result:
                    main(['ask', 'Who is Naruto?', '--kb', 'anilist'])
                self.assertEqual(result.exception.code, 1)
            else:
                main(['ask', 'Who is Naruto?', '--kb', 'anilist'])
        return output.getvalue(), errors.getvalue()

    def test_answer_alone_goes_to_stdout(self):
        output, errors = self.run_cli()
        self.assertEqual(output, 'A natural answer.\n')
        self.assertEqual(errors, '')

    def test_trace_id_goes_to_stderr_when_tracing_is_on(self):
        output, errors = self.run_cli(exporting=True)
        self.assertEqual(output, 'A natural answer.\n')
        self.assertRegex(errors, r'^Trace [0-9a-f]{32} in project ')

    def test_failed_questions_report_the_error_and_the_trace(self):
        output, errors = self.run_cli(fail=True, exporting=True)
        self.assertEqual(output, '')
        self.assertIn('Trace ', errors)
        self.assertIn('Error: Model unavailable', errors)
