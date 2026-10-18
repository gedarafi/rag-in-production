"""Capture feature candidates, plan, and evidence for the reported wording."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline
from src.recommender.entity_agent import EntityAgent

diagnostics.OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/latency-diagnostics/sad-recommendations')
diagnostics.OUT.mkdir(parents=True, exist_ok=True)
diagnostics.requests.post = diagnostics.post
pipeline = RagPipeline(load_settings('settings.toml', 'anilist'))
pipeline._agent = EntityAgent(pipeline.settings, pipeline.models, pipeline.anime_tools(), pipeline.vector_search)
agent = pipeline._agent
candidates = agent.feature_candidates('Recommend me some sad anime')
lookups = {kind: agent.store.lookup(kind, 'sad') for kind in ('genre', 'tag')}
print('CANDIDATES ' + json.dumps(candidates), flush=True)
print('LOOKUPS ' + json.dumps(lookups), flush=True)
(diagnostics.OUT / 'features.json').write_text(json.dumps({'candidates': candidates, 'lookups': lookups}, indent=2), encoding='utf-8')
reply = pipeline.ask('Recommend me some sad anime', [], debug=lambda e: print('STAGE ' + json.dumps(e), flush=True))
(diagnostics.OUT / 'route.json').write_text(json.dumps(pipeline.last_route, indent=2), encoding='utf-8')
print('REPLY ' + reply, flush=True)
