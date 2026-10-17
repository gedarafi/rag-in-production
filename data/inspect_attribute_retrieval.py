import json
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import RagPipeline
from src.config import load_settings
from src.semantic_attributes import retrieve_attributes
from src.request_resolution import PREFERENCE_STOP
from src.anime_tools import normalize

parser = argparse.ArgumentParser(description='Inspect semantic attribute retrieval; a previous capture is optional.')
parser.add_argument('question', nargs='?', default='Recommend fantasy TV anime with at most 12 episodes since 2015')
args = parser.parse_args()
p = RagPipeline(load_settings(knowledge_base='anilist'))
capture = Path('data/semantic-request-regression.json')
records = json.loads(capture.read_text(encoding='utf-8')) if capture.is_file() else []
record = next((r for r in records if args.question in r['question']), None)
q = record['question'] if record else args.question
decomposition = record['routing']['request_decomposition'] if record else {
    'entity_queries': [], 'required': [q], 'excluded': [], 'preferred': []}
catalog = p.anime_tools()
features = catalog.query('SELECT kind,id,name FROM entity_dictionary')
phrases = [v for key in ('entity_queries','required','excluded','preferred') for v in decomposition[key]
           if normalize(v) not in PREFERENCE_STOP]
print(decomposition)
print(retrieve_attributes(p.models.embed, q, phrases, catalog.searchable_fields(), features))
