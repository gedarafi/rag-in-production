"""Trace real semantic tag discovery and a single planned recommendation run."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.config import load_settings
from src.rag import RagPipeline

diagnostics.OUT = Path('data/latency-diagnostics/open-ended-tags')
diagnostics.OUT.mkdir(parents=True, exist_ok=True)
diagnostics.requests.post = diagnostics.post
p = RagPipeline(load_settings('settings.toml', 'anilist'))
question = 'Give me anime where the MC turned bad'
reply = p.ask(question, [], debug=lambda e: print('STAGE ' + json.dumps(e), flush=True))
route = p.last_route
print('RESULT ' + json.dumps({'reply': reply, 'route': route}), flush=True)
(diagnostics.OUT / 'result.json').write_text(json.dumps({'reply': reply, 'route': route}, indent=2), encoding='utf-8')
assert route['llm_calls'] == 2, route
searches = [e for e in route['agent_trace'] if e.get('tool') == 'search_features']
assert searches and searches[0]['result']['matches'], route
assert all(m['kind'] == 'tag' for m in searches[0]['result']['matches']), route
recommendations = [e for e in route['agent_trace'] if e.get('tool') == 'recommend_anime']
assert recommendations and recommendations[0]['result']['items'], route
assert recommendations[0]['arguments']['tag_ids'], route
assert 'sort_by=popularity' not in str(recommendations[0]['result']), route
