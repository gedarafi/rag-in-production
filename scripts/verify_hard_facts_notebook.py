"""Execute every evaluation cell in a real Jupyter kernel and retain verified outputs.

--max-new-questions 1 exercises one live resume; 0 checks saved checkpoints only.
The execution budget is temporary; the notebook's configured full-run budget is retained.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-new-questions', type=int, default=0)
args = parser.parse_args()
path = ROOT / 'notebook/hard_facts_evaluation.ipynb'
notebook = nbformat.read(path, as_version=4)
configuration = next(c for c in notebook.cells if c.cell_type == 'code' and c.source.startswith('RUN_LIVE ='))
original = configuration.source
configuration.source += f'\nMAX_NEW_QUESTIONS = {args.max_new_questions}\nprint("Verification execution budget: {args.max_new_questions} new live attempts; normal notebook configuration is retained.")\n'
runtime = ROOT / 'data/evaluation/hard_facts/.jupyter'
kernel = runtime / 'kernels/project-evaluation'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Project evaluation verification', 'language': 'python',
}), encoding='utf-8')
os.environ.update(JUPYTER_PATH=str(runtime), JUPYTER_RUNTIME_DIR=str(runtime / 'runtime'),
                  IPYTHONDIR=str(runtime / 'ipython'),
                  MPLBACKEND='module://matplotlib_inline.backend_inline')
client = NotebookClient(notebook, timeout=600, kernel_name='project-evaluation',
                        allow_errors=False, resources={'metadata': {'path': str(ROOT)}})
try:
    client.execute()
finally:
    configuration.source = original
errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
assert not errors, errors
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells in a real Jupyter kernel; zero error outputs.')
print('Saved notebook with verified tables and inline charts:', path)
