"""Check that the user's prepared library resumes without document reprocessing."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.anilist_setup import AniListWorkspace
from src.config import load_settings
from src.rag import RagPipeline

workspace = AniListWorkspace(load_settings('settings.toml', 'anilist'))
assert workspace.is_prepared(), 'Expected the previously verified library'
with patch.object(RagPipeline, 'ingest', side_effect=AssertionError('Unexpected reindexing')), \
     patch('src.anilist_setup.prepare_anilist', side_effect=AssertionError('Unexpected document preparation')):
    snapshots = list(workspace.setup())
result = snapshots[-1]
Path('data/latency-diagnostics/setup-reuse.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
assert result['ready'], result['message']
for key, stage in result['steps'].items():
    print(key + ': ' + stage['message'], flush=True)
