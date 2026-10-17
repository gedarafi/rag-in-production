"""Strict real-planner checks for combined metadata filters (no answer calls)."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_settings
from src.rag import RagPipeline
from src.entity_agent import EntityAgent
from src.tool_plan import plan_schema, validate_plan

p = RagPipeline(load_settings('settings.toml', 'anilist'))
agent = EntityAgent(p.settings, p.models, p.anime_tools())
cases = [
    ('Recommend good anime by Kyoto Animation before 2010',
     {'year_to': 2009, 'studio': 'Kyoto Animation', 'sort_by': 'score'}, ['year_from', 'staff']),
    ('Recommend good anime directed by Hayao Miyazaki',
     {'staff': 'Hayao Miyazaki', 'staff_role': 'Director', 'sort_by': 'score'}, ['year_from', 'year_to', 'studio']),
    ('Recommend good anime adapted from light novels between 2000 and 2020',
     {'year_from': 2000, 'year_to': 2020, 'source_material': 'LIGHT_NOVEL', 'sort_by': 'score'}, ['studio']),
    ('Recommend good anime released between 1985 and 2005',
     {'year_from': 1985, 'year_to': 2005, 'sort_by': 'score'}, ['studio', 'staff']),
]
results = []
for question, expected, absent in cases:
    _, state, evidence, current = agent.prepare(question, [])
    state['semantic_feature_candidates'] = agent.feature_candidates(question)
    for candidate in state['semantic_feature_candidates']:
        agent.register(candidate, state)
        agent.register(candidate, state['current_only'])
    schema = plan_schema(p.settings.agent_max_steps)
    plan = validate_plan(p.models.plan_tools(agent.plan_prompt(question, [], state, current, evidence, agent.guides(question)), schema), schema)
    print(json.dumps({'question': question, 'plan': plan}), flush=True)
    calls = [s['arguments'] for s in plan['steps'] if s['tool'] in ('recommend_anime', 'popular')]
    assert calls, plan
    args = calls[-1]
    for key, value in expected.items():
        actual = args.get(key)
        if key == 'source_material' and isinstance(actual, str):
            actual = actual.upper().replace(' ', '_')
        assert actual == value, (key, args)
    assert all(key not in args for key in absent), args
    # Verify the produced filters actually return records from this catalog.
    constraints = {k: args[k] for k in expected}
    items = agent.store.recommendations([], [], 5, constraints)['items']
    assert items, args
    results.append({'question': question, 'plan': plan, 'titles': [r['title'] for r in items]})
Path('data/latency-diagnostics/metadata-filters/verified-plans.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
