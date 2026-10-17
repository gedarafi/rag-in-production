import json
from pathlib import Path
from src.config import load_settings
from src.rag import RagPipeline

pipeline = RagPipeline(load_settings(knowledge_base='anilist'))
questions = [
    "Recommend me some romance anime that wasn't set in high school and are emotional",
    "I'd like an Action Anime that doesn't have romance",
    'Suggest science fiction movies released after 2010, avoid military themes, and make them thought-provoking',
]
results = []
for question in questions:
    print('RUNNING: ' + question, flush=True)
    entry = {'question': question}
    try:
        entry['answer'] = pipeline.ask(question, debug=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    except Exception as error:
        entry['error'] = str(error)
    entry['routing'] = pipeline.last_route
    results.append(entry)
    Path('data/general-request-validation-fixed.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(entry, ensure_ascii=False, indent=2), flush=True)
