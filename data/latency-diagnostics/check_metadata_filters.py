"""Live planner and local-catalog checks for timeframe and production filters."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latency_diagnostics as diagnostics
from src.config import load_settings
from src.rag import RagPipeline
from src.tool_plan import plan_schema, validate_plan

diagnostics.OUT = Path('data/latency-diagnostics/metadata-filters')
diagnostics.OUT.mkdir(parents=True, exist_ok=True)
diagnostics.requests.post = diagnostics.post
pipeline = RagPipeline(load_settings('settings.toml', 'anilist'))
index = pipeline.settings.index / 'vectors.npy'
before = (index.stat().st_size, index.stat().st_mtime_ns)
diagnostics.label_override = 'nineties'
reply = pipeline.ask('What good anime came out at the 90s', [], debug=lambda e: print('STAGE ' + json.dumps(e), flush=True))
agent = pipeline._agent
route = pipeline.last_route
print('NINETIES ' + json.dumps({'route': route, 'reply': reply}), flush=True)
evidence = json.loads((diagnostics.OUT / 'nineties-request.json').read_text(encoding='utf-8'))
results = [{'question': 'What good anime came out at the 90s', 'route': route, 'reply': reply}]
cases = [
    ('range', 'Recommend good anime released between 1985 and 2005'),
    ('studio', 'Recommend good anime by Kyoto Animation before 2010'),
    ('source', 'Recommend good anime adapted from light novels between 2000 and 2020'),
    ('staff', 'Recommend good anime directed by Hayao Miyazaki'),
]
for label, question in cases:
    diagnostics.label_override = label
    _, state, loaded, current = agent.prepare(question, [])
    state['semantic_feature_candidates'] = agent.feature_candidates(question)
    for candidate in state['semantic_feature_candidates']:
        agent.register(candidate, state)
        agent.register(candidate, state['current_only'])
    schema = plan_schema(pipeline.settings.agent_max_steps)
    plan = validate_plan(pipeline.models.plan_tools(agent.plan_prompt(
        question, [], state, current, loaded, agent.guides(question)), schema), schema)
    print('PLAN ' + json.dumps({'question': question, 'plan': plan}), flush=True)
    results.append({'question': question, 'plan': plan})
    assert plan['scope'] != 'out_of_domain'
    calls = [s['arguments'] for s in plan['steps'] if s['tool'] in ('recommend_anime', 'popular')]
    assert calls, plan
    filters = calls[-1]
    if label == 'range':
        assert filters.get('year_from') == 1985 and filters.get('year_to') == 2005, plan
    elif label == 'studio':
        assert filters.get('studio') == 'Kyoto Animation' and filters.get('year_to') == 2009, plan
        assert 'year_from' not in filters, plan
    elif label == 'source':
        assert filters.get('source_material', '').upper().replace(' ', '_') == 'LIGHT_NOVEL', plan
    elif label == 'staff':
        assert filters.get('staff') == 'Hayao Miyazaki' and filters.get('staff_role') == 'Director', plan
        assert not any(k in filters for k in ('year_from', 'year_to', 'studio')), plan
# Independently enforce every metadata combination against actual catalog data.
for filters in ({'year_from': 1990, 'year_to': 1999, 'sort_by': 'score'},
                {'year_from': 1985, 'year_to': 2005, 'sort_by': 'score'},
                {'studio': 'Kyoto Animation', 'year_to': 2009, 'sort_by': 'score'},
                {'source_material': 'LIGHT_NOVEL', 'year_from': 2000, 'year_to': 2020, 'sort_by': 'score'},
                {'staff': 'Hayao Miyazaki', 'staff_role': 'Director', 'sort_by': 'score'}):
    items = agent.store.recommendations([], [], 5, filters)['items']
    assert items, filters
    assert all('year_from' not in filters or r['year'] >= filters['year_from'] for r in items)
    assert all('year_to' not in filters or r['year'] <= filters['year_to'] for r in items)
    assert [r['score'] for r in items] == sorted((r['score'] for r in items), reverse=True)
    print('FILTER ' + json.dumps({'filters': filters, 'titles': [r['title'] for r in items]}), flush=True)
assert before == (index.stat().st_size, index.stat().st_mtime_ns), 'Vector index changed'
(diagnostics.OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
