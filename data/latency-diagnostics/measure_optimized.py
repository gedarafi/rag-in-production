"""Measure the optimized application using the same transport instrumentation."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.config import load_settings
from src.rag import RagPipeline

diagnostics.OUT = Path('data/latency-diagnostics/optimized')
diagnostics.OUT.mkdir(parents=True, exist_ok=True)
diagnostics.requests.post = diagnostics.post
pipeline = RagPipeline(load_settings('settings.toml', 'anilist'))
events = []
def debug(event):
    events.append(event)
    print('STAGE ' + json.dumps(event), flush=True)
reply = pipeline.ask('What was Clannad about?', [], debug=debug)
(diagnostics.OUT / 'stages.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
(diagnostics.OUT / 'route.json').write_text(json.dumps(pipeline.last_route, indent=2), encoding='utf-8')
print('ANSWER ' + reply, flush=True)
# Validate the old reply against the original full evidence to isolate the regex
# optimization from both changed model output and changed prompt selection.
base = Path('data/latency-diagnostics')
payload = json.loads((base / 'answer-request.json').read_text(encoding='utf-8'))
content = payload['messages'][1]['content']
data, _ = json.JSONDecoder().raw_decode(content[content.index('{'):])
records = json.loads((base / 'metrics.json').read_text(encoding='utf-8'))
old_reply = next(r['response'] for r in records if r['label'] == 'answer')
started = time.perf_counter()
issues = pipeline._agent.identity_issues(old_reply, data['evidence'])
validation = {'seconds': time.perf_counter() - started, 'issues': issues}
(diagnostics.OUT / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
print('VALIDATION ' + json.dumps(validation), flush=True)
