"""Execute the two published notebooks offline without changing their originals."""
import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Directory for verified copies; original notebooks are untouched')
    parser.add_argument('--source-only', action='store_true', help='Copy Git-visible source files only, excluding all ignored runtime data')
    parser.add_argument('--tests', action='store_true', help='Also execute the unit suite in the selected checkout')
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / 'data/notebook-output-backups/offline-verification').resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.source_only:
        source = root
        root = Path(tempfile.mkdtemp(prefix='source-checkout-', dir=output))
        names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=source).decode().split('\0')
        for name in set(names) - {''}:
            path = source / name
            if path.is_file():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        assert not (root / 'data/evaluation/recommendations/catalog_reference.jsonl').exists()
        assert not (root / 'data/documents/anilist').exists()
        assert not (root / 'data/index').exists()
        print('Source-only checkout:', root, flush=True)
    if args.tests:
        log = output / 'tests.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/recommender', '-t', '.', '-v'], cwd=root,
                                    stdout=stream, stderr=subprocess.STDOUT)
        print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-5:]), flush=True)
        result.check_returncode()
    runtime = output / '.jupyter'
    kernel = runtime / 'kernels/source-only-check'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Offline source verification', 'language': 'python',
    }), encoding='utf-8')
    os.environ.update(JUPYTER_PATH=str(runtime), JUPYTER_RUNTIME_DIR=str(runtime / 'runtime'),
                      IPYTHONDIR=str(runtime / 'ipython'))
    report = []
    for name in ('hard_facts_evaluation.ipynb', 'recommendation_evaluation.ipynb'):
        path = root / 'notebook' / name
        notebook = nbformat.read(path, as_version=4)
        original_sources = {}
        for cell in notebook.cells:
            if cell.cell_type == 'code' and 'RUN_LIVE =' in cell.source:
                original_sources[cell.id] = cell.source
                cell.source += '\nRUN_LIVE = False\n'
        NotebookClient(notebook, timeout=180, kernel_name='source-only-check', allow_errors=False,
                       resources={'metadata': {'path': str(root / 'notebook')}}).execute()
        for cell in notebook.cells:
            if cell.id in original_sources:
                cell.source = original_sources[cell.id]
        nbformat.validate(notebook)
        nbformat.write(notebook, output / path.name)
        report.append({'notebook': path.name, 'code_cells': sum(c.cell_type == 'code' for c in notebook.cells),
                       'status': 'passed'})
        print(path.name, 'passed', flush=True)
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
