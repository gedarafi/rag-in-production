import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from src.recommender.cli import main


class FakeWorkspace:
    def __init__(self, fail):
        self.fail = fail

    def setup(self, source, rebuild, progress=None):
        status = 'error' if self.fail else 'completed'
        step = {'label': 'Document embeddings', 'status': status, 'message': 'embedding interrupted' if self.fail else 'Reusing the index.'}
        yield {'ready': False, 'message': 'Document embeddings…', 'steps': {'embeddings': {**step, 'status': 'running'}}}
        yield {'ready': not self.fail, 'message': step['message'], 'steps': {'embeddings': step}}


class CliSetupTests(unittest.TestCase):
    def run_setup(self, fail):
        output, errors = io.StringIO(), io.StringIO()
        with patch('sys.argv', ['recommender', 'setup']), \
             patch('src.recommender.anilist_setup.AniListWorkspace', return_value=FakeWorkspace(fail)), \
             redirect_stdout(output), redirect_stderr(errors):
            try:
                main()
                code = 0
            except SystemExit as exit:
                code = exit.code
        return code, output.getvalue(), errors.getvalue()

    def test_setup_prints_each_stage_once_and_reports_ready(self):
        code, output, _ = self.run_setup(fail=False)
        self.assertEqual(code, 0)
        self.assertEqual(output.count('Document embeddings: Reusing the index.'), 1)
        self.assertIn('Library ready', output)

    def test_failed_stage_exits_nonzero(self):
        code, output, errors = self.run_setup(fail=True)
        self.assertEqual(code, 1)
        self.assertIn('✕ Document embeddings: embedding interrupted', output)
        self.assertIn('embedding interrupted', errors)


if __name__ == '__main__':
    unittest.main()
