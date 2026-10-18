import json
from pathlib import Path
from src.recommender.config import load_settings
from src.recommender.rag import RagPipeline

pipeline = RagPipeline(load_settings(knowledge_base='anilist'))
questions = ['When did One-Punch Man come out?', 'How many episodes does One-Punch Man have, and how long is each episode?',
             'Recommend fantasy TV anime with at most 12 episodes released since 2015, preferably funny']
results = []
for question in questions:
    print('RUNNING: ' + question, flush=True)
    entry = {'question': question}
    try:
        entry['answer'] = pipeline.ask(question, debug=lambda event: print(json.dumps(event), flush=True))
    except Exception as error:
        entry['error'] = str(error)
    entry['routing'] = pipeline.last_route
    results.append(entry)
    Path('data/attribute-request-validation.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(entry, ensure_ascii=False, indent=2), flush=True)
