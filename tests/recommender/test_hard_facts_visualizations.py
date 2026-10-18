import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ChartEnvironmentTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Install the evaluation extra for chart tests')
    def test_export_with_inherited_unavailable_notebook_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / 'questions.csv'
            with questions.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['question_id', 'anime_id', 'field'])
                writer.writeheader()
                writer.writerow({'question_id': 'q1', 'anime_id': '20', 'field': 'year'})
            output = root / 'charts'
            result = subprocess.run(
                [sys.executable, '-m', 'src.recommender.hard_facts_visualizations', '--questions', str(questions), '--output', str(output)],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, 'MPLBACKEND': 'module://unavailable_notebook_backend'},
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            charts = json.loads((output / 'charts.json').read_text(encoding='utf-8'))
            self.assertEqual(len(charts), 1)
            for extension in ('png', 'svg'):
                self.assertGreater(Path(charts[0][extension]).stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
