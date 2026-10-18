"""Notebook source maintenance must preserve existing saved execution output."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('notebook_utils', ROOT / 'scripts/notebook_utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class NotebookSourceTests(unittest.TestCase):
    def test_source_update_retains_saved_outputs_by_cell_identity(self):
        output = {'output_type': 'display_data', 'data': {'text/plain': 'saved chart'}, 'metadata': {}}
        original = {'cells': [{'id': 'one', 'cell_type': 'code', 'source': 'print(1)',
                               'outputs': [output], 'execution_count': 7}], 'metadata': {}}
        revised = {'cells': [{'id': 'one', 'cell_type': 'code', 'source': 'print(2)',
                              'outputs': [], 'execution_count': None}], 'metadata': {}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.ipynb'
            path.write_text(json.dumps(original), encoding='utf-8')
            utils.write_notebook(path, revised)
            result = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(result['cells'][0]['source'], 'print(2)')
            self.assertEqual(result['cells'][0]['outputs'], [output])
            self.assertEqual(result['cells'][0]['execution_count'], 7)
