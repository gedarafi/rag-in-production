"""Trace generic evidence loading and reasoning for a comparison question."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline

diagnostics.OUT = Path('data/latency-diagnostics/multi-title-facts')
diagnostics.OUT.mkdir(parents=True, exist_ok=True)
diagnostics.requests.post = diagnostics.post
p = RagPipeline(load_settings('settings.toml', 'anilist'))
question = 'What are the similarity between Naruto and JJK?'
reply = p.ask(question, [], debug=lambda e: print('STAGE ' + json.dumps(e), flush=True))
route = p.last_route
result = {'reply': reply, 'route': route}
print('RESULT ' + json.dumps(result), flush=True)
(diagnostics.OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
assert 'naruto' in reply.lower() and ('jujutsu' in reply.lower() or 'jjk' in reply.lower()), reply
assert route['llm_calls'] == 2, route
assert not any(row.get('tool') in ('search_features', 'recommend_anime', 'recommend_by_character', 'popular')
               for row in route['agent_trace']), route
request = json.loads((diagnostics.OUT / 'answer-request.json').read_text(encoding='utf-8'))
prompt = request['messages'][-1]['content']
payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"history":'):])
assert {row['id'] for row in payload['primary_entity_facts'] if row['kind'] == 'anime'} == {'20', '113415'}, payload
