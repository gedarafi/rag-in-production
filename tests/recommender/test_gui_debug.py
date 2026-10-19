import io
import json
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from src.recommender.gui import build_app


class GuiDebugTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = SimpleNamespace(last_route={'llm_calls': 2, 'use_previous_context': True,
                                                   'router_plan': {'steps': []}},
                                        ask=lambda *args, **kwargs: 'A natural answer.')
        with patch('src.recommender.gui.RagPipeline', return_value=self.pipeline):
            self.app = build_app()
            self.ask = next(f.fn for f in self.app.fns.values() if f.fn and f.fn.__name__ == 'ask_question')
            # Pipeline creation is lazy, so advance the generator within the patch.
            self.result = list(self.ask('A question', [], 'anilist', {}))[-1]

    def test_debug_is_hidden_by_default_but_snapshot_is_session_scoped(self):
        panel = next(c for c in self.app.config['components'] if c['props'].get('label') == 'Router debug')
        self.assertFalse(panel['props']['visible'])
        self.assertNotIn('visible', self.result[4])
        self.assertEqual(self.result[3]['llm_calls'], 2)
        self.pipeline.last_route['llm_calls'] = 99
        self.assertEqual(self.result[3]['llm_calls'], 2)

    def test_debug_toggle_uses_saved_snapshot_without_another_query(self):
        toggle = next(f.fn for f in self.app.fns.values() if f.fn and len(f.inputs) == 2 and len(f.outputs) == 1
                      and f.fn.__name__ == '<lambda>')
        self.assertTrue(toggle(True, self.result[3])['visible'])
        self.assertFalse(toggle(False, self.result[3])['visible'])

    def test_failed_request_includes_debug_information(self):
        def fail(*args, **kwargs):
            raise RuntimeError('Model unavailable')
        self.pipeline.ask = fail
        terminal = io.StringIO()
        with redirect_stderr(terminal):
            result = list(self.ask('Another question', [], 'anilist', {}))[-1]
        self.assertEqual(result[3]['status'], 'error')
        self.assertNotIn('visible', result[4])
        self.assertIn('Model unavailable', result[3]['error'])
        self.assertIn('"status": "error"', terminal.getvalue())
        self.assertIn('Model unavailable', terminal.getvalue())
        self.assertGreaterEqual(result[3]['elapsed_seconds'], 0)

    def test_terminal_debug_includes_start_and_complete_snapshot(self):
        terminal = io.StringIO()
        with redirect_stderr(terminal):
            result = list(self.ask('Who are the characters?', [], 'anilist', {}))[-1]
        output = terminal.getvalue()
        self.assertIn('"status": "running"', output)
        self.assertGreaterEqual(result[3]['elapsed_seconds'], 0)
        self.assertIn('"status": "completed"', output)
        self.assertIn('"llm_calls": 2', output)

    def test_snapshot_names_the_phoenix_trace(self):
        self.pipeline.last_trace_id = 'a' * 32
        terminal = io.StringIO()
        with redirect_stderr(terminal):
            result = list(self.ask('A question', [], 'anilist', {}))[-1]
        self.assertEqual(result[3]['trace_id'], 'a' * 32)
        self.assertIn('"trace_id": "' + 'a' * 32, terminal.getvalue())

if __name__ == '__main__':
    unittest.main()
