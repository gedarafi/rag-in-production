"""Execute the recommendation notebook offline in the actual project kernel."""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'notebook/recommendation_evaluation.ipynb'
notebook = nbformat.read(path, as_version=4)
configuration = next(c for c in notebook.cells if c.cell_type == 'code' and c.source.startswith('RUN_LIVE ='))
original = configuration.source
configuration.source += '\nRUN_LIVE = False\n'
runtime = ROOT / 'data/evaluation/recommendations/.jupyter'
kernel = runtime / 'kernels/project-recommendations'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Project recommendation verification', 'language': 'python',
}), encoding='utf-8')
os.environ.update(JUPYTER_PATH=str(runtime), JUPYTER_RUNTIME_DIR=str(runtime / 'runtime'),
                  IPYTHONDIR=str(runtime / 'ipython'))
client = NotebookClient(notebook, timeout=300, kernel_name='project-recommendations', allow_errors=False,
                        resources={'metadata': {'path': str(ROOT / 'notebook')}})
try:
    client.execute()
finally:
    configuration.source = original
nbformat.validate(notebook)
assert not [o for c in notebook.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
nbformat.write(notebook, path)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells with zero errors; outputs saved to {path}')
