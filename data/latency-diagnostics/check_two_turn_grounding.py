"""Record the reported multi-work question and its referential follow-up."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline

diagnostics.requests.post = diagnostics.post
p = RagPipeline(load_settings('settings.toml', 'anilist'))
history = []
run = sys.argv[1] if len(sys.argv) > 1 else 'before'
questions = ['How long is Naruto and Naruto Shippuden?', 'how many episodes it is?']
if run == 'recovery':
    history = [{'role': 'user', 'content': questions[0]},
               {'role': 'assistant', 'content': 'The total duration is 1030 minutes.'}]
    questions = questions[1:]
for number, question in enumerate(questions):
    diagnostics.OUT = Path('data/latency-diagnostics/two-turn-grounding') / run / str(number)
    diagnostics.OUT.mkdir(parents=True, exist_ok=True)
    diagnostics.records.clear()
    reply = p.ask(question, history, debug=lambda e: print('STAGE ' + json.dumps(e), flush=True))
    result = {'reply': reply, 'route': p.last_route}
    (diagnostics.OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULT ' + json.dumps(result), flush=True)
    history.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}])
    if run != 'before':
        assert p.last_route['llm_calls'] == 2, result
        assert {row['id'] for row in p.last_route['selected_context']['anime']} == {'20', '1735'}, result
        if 'episodes' in question:
            assert '220' in reply and '500' in reply, result
            assert '12 episodes' not in reply and '13 episodes' not in reply, result
