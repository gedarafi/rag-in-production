import json
import sys
from pathlib import Path
from src.config import load_settings
from src.rag import RagPipeline

pipeline = RagPipeline(load_settings(knowledge_base='anilist'))
cases = [
    ("Recommend me some romance anime that wasn't set in high school and are emotional",
     [('genres', 'eq', 'Romance'), ('tags', 'ne', 'School')]),
    ("I'd like an Action Anime that doesn't have romance", [('genres', 'eq', 'Action'), ('genres', 'ne', 'Romance')]),
    ('Suggest science fiction movies released after 2010, avoid military themes, and make them thought-provoking',
     [('genres', 'eq', 'Sci-Fi'), ('format', 'eq', 'MOVIE'), ('year', 'gt', '2010'), ('tags', 'ne', 'Military')]),
    ('When did One-Punch Man come out?', []),
    ('How many episodes does One-Punch Man have, and how long is each episode?', []),
    ('Recommend fantasy TV anime with at most 12 episodes released since 2015, preferably funny',
     [('genres', 'eq', 'Fantasy'), ('format', 'eq', 'TV'), ('episodes', 'lte', '12'), ('year', 'gte', '2015')]),
]
selected = set(map(int, sys.argv[1:])) if len(sys.argv) > 1 else set(range(len(cases)))
path = Path('data/final-request-regression.json')
results = json.loads(path.read_text(encoding='utf-8')) if len(sys.argv) > 1 and path.is_file() else []
for index, (question, expected) in enumerate(cases):
    if index not in selected:
        continue
    print('RUNNING: ' + question, flush=True)
    entry = {'question': question}
    try:
        entry['answer'] = pipeline.ask(question, debug=lambda event: print(json.dumps(event), flush=True))
        request = pipeline.last_route.get('request_constraints', {})
        actual = {(f['field'], f['operator'], f['value']) for f in request.get('filters', [])}
        assert set(expected) == actual, f'Constraint mismatch: missing={set(expected) - actual}, extra={actual - set(expected)}'
        if expected:
            traces = [r for r in pipeline.last_route['agent_trace'] if r.get('result', {}).get('items')]
            assert traces, 'No recommendation candidates'
            for trace in traces:
                for item in trace['result']['items']:
                    aid = item['id']
                    clauses, params = pipeline.anime_tools()._where({'filters': request['filters']})
                    assert pipeline.anime_tools().query('SELECT a.id FROM anime a WHERE a.id=? AND ' +
                                                       clauses, [aid, *params]), 'Candidate violates filters'
        else:
            expected_numbers = ['2015'] if 'come out' in question else ['12', '24']
            assert all(number in entry['answer'] for number in expected_numbers), 'Incorrect factual answer'
        entry['validation'] = 'PASS'
    except Exception as error:
        entry['validation'] = 'FAIL'
        entry['error'] = str(error)
    entry['routing'] = pipeline.last_route
    results = [row for row in results if row['question'] != question] + [entry]
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'question': question, 'validation': entry['validation'], 'error': entry.get('error')}, ensure_ascii=False), flush=True)
