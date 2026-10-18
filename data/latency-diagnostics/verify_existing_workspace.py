"""Verify already-built local artifacts and restore a missing readiness marker."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.recommender.anilist_setup import AniListWorkspace
from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline

workspace = AniListWorkspace(load_settings('settings.toml', 'anilist'))
workspace.pipeline = RagPipeline(workspace.settings)
print(workspace.verify(), flush=True)
temporary = workspace.manifest.with_suffix('.json.tmp')
temporary.write_text(json.dumps(workspace.signature(), indent=2), encoding='utf-8')
temporary.replace(workspace.manifest)
assert AniListWorkspace(workspace.settings).ready
print('Existing workspace verified; chat ready without rebuilding.', flush=True)
