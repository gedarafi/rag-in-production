import json
import sys
from src.rag import RagPipeline
from src.config import load_settings
from src.semantic_attributes import retrieve_attributes
from src.request_resolution import PREFERENCE_STOP
from src.anime_tools import normalize

p = RagPipeline(load_settings(knowledge_base='anilist'))
records = json.load(open('data/semantic-request-regression.json', encoding='utf-8'))
record = next(r for r in records if sys.argv[1] in r['question'])
q, decomposition = record['question'], record['routing']['request_decomposition']
catalog = p.anime_tools()
features = catalog.query('SELECT kind,id,name FROM entity_dictionary')
phrases = [v for key in ('entity_queries','required','excluded','preferred') for v in decomposition[key]
           if normalize(v) not in PREFERENCE_STOP]
print(decomposition)
print(retrieve_attributes(p.models.embed, q, phrases, catalog.searchable_fields(), features))
