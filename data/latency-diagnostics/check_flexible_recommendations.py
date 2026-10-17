"""Exercise vague preferences through the real planner and answer model."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.config import load_settings
from src.rag import RagPipeline

diagnostics.requests.post = diagnostics.post
p = RagPipeline(load_settings('settings.toml', 'anilist'))
for name, question in [('short', 'Recommend me a short anime'), ('beginner', 'Recommend me an anime for beginners')]:
    diagnostics.OUT = Path('data/latency-diagnostics/flexible-recommendations') / name
    diagnostics.OUT.mkdir(parents=True, exist_ok=True)
    diagnostics.records.clear()
    reply = p.ask(question, [], debug=lambda event: print('STAGE ' + json.dumps(event), flush=True))
    result = {'question': question, 'reply': reply, 'route': p.last_route}
    (diagnostics.OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULT ' + json.dumps(result), flush=True)
    recommendations = [r for r in p.last_route['agent_trace'] if r.get('tool') in ('recommend_anime', 'popular')]
    assert recommendations and recommendations[-1].get('result', {}).get('items'), result
    assert p.last_route['scope'] == 'in_domain' and p.last_route['llm_calls'] == 2, result
