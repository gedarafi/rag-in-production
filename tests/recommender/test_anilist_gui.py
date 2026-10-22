import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.recommender.anilist_gui import build_app


class AniListGuiTests(unittest.TestCase):
    def test_app_is_anilist_only_and_chat_is_gated_until_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'settings.toml'
            config.write_text(Path('settings.toml').read_text(encoding='utf-8'), encoding='utf-8')
            original = config.read_bytes()
            app = build_app(str(config))
            components = app.config['components']
            self.assertFalse(any(c['type'] == 'dropdown' for c in components))
            send = next(c for c in components if c['type'] == 'button' and c['props'].get('value') == 'Send')
            self.assertFalse(send['props']['interactive'])
            setup = next(c for c in components if c['type'] == 'accordion' and c['props'].get('label') == 'Library setup')
            self.assertTrue(setup['props']['open'])
            self.assertEqual(config.read_bytes(), original)
            ask = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == 'ask_question')
            with patch('src.recommender.anilist_gui.print_debug'):
                result = list(ask('Who are the characters?', [], {}))[-1]
            self.assertEqual(result[3]['status'], 'error')
            self.assertIn('Run setup', result[3]['error'])
            # The answer's position in the chat maps to the question ID that feedback is logged with.
            self.assertEqual(result[5], {1: result[3]['question_id']})
            feedback = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == 'record_feedback')
            with patch('src.recommender.anilist_gui.print_debug') as logged:
                feedback(SimpleNamespace(index=1, liked=False), result[5])
            logged.assert_called_once_with({'app': 'anilist', 'feedback': {
                'question_id': result[3]['question_id'], 'message_index': 1, 'rating': 'down'}})


if __name__ == '__main__':
    unittest.main()
