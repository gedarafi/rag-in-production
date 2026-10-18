"""Profile post-generation validation using a captured reply; no model call."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline
from src.recommender.entity_agent import EntityAgent
from capture_support import ensure_captures

out = ensure_captures(['answer-request.json', 'metrics.json', 'answer-compact-request.json'])
settings = load_settings('settings.toml', 'anilist')
agent = EntityAgent(settings, None, RagPipeline(settings).anime_tools())
request = json.loads((out / 'answer-request.json').read_text(encoding='utf-8'))
text = request['messages'][1]['content']
data, _ = json.JSONDecoder().raw_decode(text[text.index('{'):])
metrics = json.loads((out / 'metrics.json').read_text())
reply = next(r['response'] for r in metrics if r['label'] == 'answer')
profiler = cProfile.Profile()
started = time.perf_counter()
issues = profiler.runcall(agent.identity_issues, reply, data['evidence'])
elapsed = time.perf_counter() - started
stream = io.StringIO()
pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(18)
(out / 'validation-profile.txt').write_text(stream.getvalue(), encoding='utf-8')
summary = {'identity_validation_seconds': elapsed, 'issues': issues,
           'evidence_chars': len(json.dumps(data['evidence'], ensure_ascii=False)),
           'history_chars': len(json.dumps(data['history'], ensure_ascii=False)),
           'history_messages': len(data['history'])}
(out / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary), flush=True)
print(stream.getvalue(), flush=True)
# Profiler overhead is substantial; separately record normal execution times.
normal = []
for label in ('answer', 'answer-compact'):
    payload = json.loads((out / (label + '-request.json')).read_text(encoding='utf-8'))
    content = payload['messages'][1]['content']
    fields, _ = json.JSONDecoder().raw_decode(content[content.index('{'):])
    started = time.perf_counter()
    issues = agent.identity_issues(reply, fields['evidence'])
    normal.append({'label': label, 'seconds': time.perf_counter() - started, 'issues': issues})
(out / 'validation-unprofiled.json').write_text(json.dumps(normal, indent=2), encoding='utf-8')
print('UNPROFILED ' + json.dumps(normal), flush=True)
