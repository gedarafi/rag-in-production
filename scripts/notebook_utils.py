"""Update notebook sources without discarding the user's saved execution outputs."""
from copy import deepcopy
import json
from pathlib import Path


def write_notebook(path, notebook):
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        old = json.loads(path.read_text(encoding='utf-8'))
        previous = {cell.get('id'): cell for cell in old['cells'] if cell.get('id')}
        for cell in notebook['cells']:
            prior = previous.get(cell.get('id'))
            if cell['cell_type'] == 'code' and prior and prior['cell_type'] == 'code':
                cell['outputs'] = deepcopy(prior.get('outputs', []))
                cell['execution_count'] = prior.get('execution_count')
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
