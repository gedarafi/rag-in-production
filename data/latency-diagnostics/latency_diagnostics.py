"""One-off local latency measurements; does not change application behavior."""
import json
from pathlib import Path
import sys
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path('data/latency-diagnostics')
OUT.mkdir(parents=True, exist_ok=True)
records = []
real_post = requests.post

def save():
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

def post(url, **kwargs):
    payload = kwargs.get('json', {})
    label = globals().get('label_override') or ('planning' if 'format' in payload else 'answer')
    (OUT / (label + '-request.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    started = time.perf_counter()
    response = real_post(url, **kwargs)
    wall = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    record = {'label': label, 'wall_seconds': wall,
              'message_chars': [len(m['content']) for m in payload.get('messages', [])],
              'schema_chars': len(json.dumps(payload.get('format', {}))),
              **{k: body.get(k) for k in ('total_duration', 'load_duration', 'prompt_eval_count',
                  'prompt_eval_duration', 'eval_count', 'eval_duration', 'done_reason')}}
    for key in ('total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'):
        record[key + '_seconds'] = (body.get(key) or 0) / 1e9
    record['outside_server_seconds'] = wall - record['total_duration_seconds']
    record['response'] = body.get('message', {}).get('content')
    records.append(record)
    save()
    print(json.dumps(record), flush=True)
    return response

if __name__ == '__main__':
    # Cheap endpoints isolate address-resolution/connection costs from inference.
    connections = []
    for host in ('localhost', '127.0.0.1'):
        for repeat in range(3):
            started = time.perf_counter()
            response = requests.get(f'http://{host}:11434/api/version', timeout=15)
            response.raise_for_status()
            connections.append({'host': host, 'seconds': time.perf_counter() - started,
                                'version': response.json()})
    (OUT / 'connections.json').write_text(json.dumps(connections, indent=2), encoding='utf-8')
    print('CONNECTIONS ' + json.dumps(connections), flush=True)
    from src.config import load_settings
    from src.rag import RagPipeline
    settings = load_settings('settings.toml', 'anilist')
    pipeline = RagPipeline(settings)
    requests.post = post
    stages = []
    def debug(event):
        stages.append(event)
        print('STAGE ' + json.dumps(event), flush=True)
    # Exact current code, explicit fresh history; original ten message bodies are unavailable.
    result = pipeline.ask('What was Clannad about?', [], debug=debug)
    (OUT / 'stages.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')
    (OUT / 'route.json').write_text(json.dumps(pipeline.last_route, indent=2), encoding='utf-8')
    for name in ('planning', 'answer'):
        payload = json.loads((OUT / (name + '-request.json')).read_text(encoding='utf-8'))
        content = payload['messages'][1]['content']
        start = content.index('{')
        data, size = json.JSONDecoder().raw_decode(content[start:])
        breakdown = {k: len(json.dumps(v, ensure_ascii=False)) for k, v in data.items()}
        (OUT / (name + '-breakdown.json')).write_text(json.dumps({'instruction_chars': start,
            'fields_chars': breakdown, 'data': data}, ensure_ascii=False, indent=2), encoding='utf-8')
        if name == 'planning':
            data.pop('retrieved_guides', None)
            data.pop('current_message_entity_ids', None)
            # Retain IDs and explicit evidence availability, omit full source content during routing.
            for item in data.get('evidence', []):
                r = item.get('result', {})
                item['result'] = {k: v for k, v in r.items() if k in ('kind', 'id', 'title', 'name', 'status')}
                item['result']['available_fields'] = list(r)
        else:
            # An explicit synopsis question needs its synopsis and identity, not cast/relations.
            seen = set()
            evidence = []
            for item in data.get('evidence', []):
                r = item.get('result', {})
                key = (r.get('kind'), r.get('id'))
                if key in seen:
                    continue
                seen.add(key)
                item['result'] = {k: v for k, v in r.items() if k in
                    ('kind', 'id', 'title', 'name', 'aliases', 'description', 'synopsis', 'source')}
                evidence.append(item)
            data['evidence'] = evidence
            data['primary_entity_facts'] = [{k: v for k, v in r.items() if k in ('kind', 'id', 'title', 'name')}
                                           for r in data.get('primary_entity_facts', [])]
        payload['messages'][1]['content'] = content[:start] + json.dumps(data, ensure_ascii=False) + content[start + size:]
        label_override = name + '-compact'
        # Keep all instruction text, schema, model, context size and generation options identical.
        post(settings.ollama_url + '/api/chat', json=payload, timeout=180)
