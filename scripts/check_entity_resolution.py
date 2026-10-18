"""Run real-model mention classification regressions without answer generation (model from MODEL_ID/MODEL_API)."""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommender.anime_tools import AnimeTools
from src.recommender.config import load_settings
from src.recommender.entity_agent import EntityAgent
from src.recommender.entity_resolution import resolve_mentions
from src.recommender.providers import Models
from src.recommender.request_resolution import interpret

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', help='Run only questions containing this text')
parser.add_argument('--output', default='data/evaluation/hard_facts/diagnostics/entity_resolution_live.json')
args = parser.parse_args()

settings = load_settings(ROOT / 'recommender.toml', 'anilist')
models = Models()
agent = EntityAgent(settings, models, AnimeTools.load(settings.documents))
cases = [
    ('When did Cowboy Bebop first air?', {('anime', '1')}, None),
    ('What is Air about?', {('anime', '101')}, None),
    ('What is Original about?', {('anime', '103762')}, None),
    ('What year did the original Naruto anime begin?', {('anime', '20')}, None),
    ("About how many minutes is one episode of Demon Slayer's first season?", {('anime', '101922')}, None),
    ('What year did Fullmetal Alchemist: Brotherhood first air?', {('anime', '5114')}, None),
    ('Recommend horror anime.', set(), ('genre', 'required')),
    ('Recommend anime without Time Loop.', set(), ('tag', 'excluded')),
    ('Who animated Monster?', {('anime', '19')}, None),
]
if args.only:
    cases = [case for case in cases if args.only.casefold() in case[0].casefold()]
results = []
for question, expected, feature in cases:
    agent.last_usage = {'llm_calls': 0}
    try:
        # Factual cases exercise the complete production interpreter. Recommendation
        # cases isolate classification/linking so optional preference mapping cannot
        # obscure whether the entity/type decision was correct.
        if feature is None:
            request = interpret(agent, question, [])
            resolution = request['entity_resolution']
            decomposition = {k: request[k] for k in ('intent', 'use_previous_context', 'entity_queries')}
        else:
            from src.recommender.mention_interpreter import classify
            decomposition, mentions = classify(agent, question, [])
            resolution = resolve_mentions(agent, question, [], decomposition, typed_mentions=mentions)
        selected = {(s['kind'], s['id']) for s in resolution['subjects']}
        feature_ok = feature is None or any((m['kind'], m['role']) == feature for m in resolution['mentions'])
        ok = selected == expected and feature_ok
        result = {'question': question, 'passed': ok, 'decomposition': decomposition, 'resolution': resolution}
    except Exception as error:
        result = {'question': question, 'passed': False, 'error': f'{type(error).__name__}: {error}',
                  'raw_resolution': agent.last_usage.get('entity_resolution_outputs', []),
                  'raw_classification': agent.last_usage.get('request_interpretation_outputs', [])}
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"Passed {sum(r['passed'] for r in results)}/{len(results)} live entity-resolution cases.")
sys.exit(0 if all(r['passed'] for r in results) else 1)
