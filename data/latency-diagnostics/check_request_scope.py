"""Exercise scope decisions with the real planner, without unrelated tool work."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.config import load_settings
from src.rag import RagPipeline
from src.tool_plan import plan_schema, validate_plan

diagnostics.OUT = Path('data/latency-diagnostics/request-scope')
diagnostics.OUT.mkdir(parents=True, exist_ok=True)
diagnostics.requests.post = diagnostics.post
pipeline = RagPipeline(load_settings('settings.toml', 'anilist'))
diagnostics.label_override = 'apple'
reply = pipeline.ask('How to cook an apple?', [])
route = pipeline.last_route
assert route['llm_calls'] == 1, route
assert route['executed_tools'] == 0, route
assert reply.strip()
results = [{'question': 'How to cook an apple?', 'scope': route['scope'],
            'reply': reply, 'llm_calls': route['llm_calls']}]
print('APPLE ' + json.dumps(results[0]), flush=True)
agent = pipeline._agent
cases = [
    ('cooking-anime', 'Recommend some cooking anime', []),
    ('summary', 'What was Clannad about?', []),
    ('unknown-title', 'What was Clannnad about?', []),
    ('followup', 'Why did she cook the apple?', [
        {'role': 'user', 'content': 'What is Food Wars about?'},
        {'role': 'assistant', 'content': 'We were discussing an episode where a girl cooked an apple.'}]),
]
for label, question, history in cases:
    diagnostics.label_override = label
    _, state, evidence, current = agent.prepare(question, history)
    state['semantic_feature_candidates'] = agent.feature_candidates(question)
    for candidate in state['semantic_feature_candidates']:
        agent.register(candidate, state)
        agent.register(candidate, state['current_only'])
    prompt = agent.plan_prompt(question, history, state, current, evidence, agent.guides(question))
    schema = plan_schema(pipeline.settings.agent_max_steps)
    plan = validate_plan(pipeline.models.plan_tools(prompt, schema), schema)
    results.append({'question': question, 'plan': plan})
    print('SCOPE ' + json.dumps(results[-1]), flush=True)
    assert plan['scope'] != 'out_of_domain', plan
(diagnostics.OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
