"""Regenerate diagnostic captures when a source checkout has no saved runtime data."""
from pathlib import Path
import subprocess
import sys


def ensure_captures(names):
    folder = Path(__file__).resolve().parent
    if not all((folder / name).is_file() for name in names):
        print('No saved diagnostic baseline. Capturing a fresh run with the configured local model.', flush=True)
        print('AniList setup and Ollama are required, as for other live diagnostics.', flush=True)
        subprocess.run([sys.executable, str(folder / 'latency_diagnostics.py')], cwd=folder.parents[1], check=True)
    missing = [name for name in names if not (folder / name).is_file()]
    if missing:
        raise RuntimeError('This model run did not produce the required diagnostic captures: ' + ', '.join(missing))
    return folder
