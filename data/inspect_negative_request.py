import json
from pathlib import Path
from src.recommender.anime_tools import AnimeTools
from src.recommender.entity_store import EntityStore

question = "I'd like an Action Anime that doesn't have romance"
catalog = AnimeTools.load(Path('data/documents/anilist'))
store = EntityStore(catalog)
result = {
    'title_candidates': catalog.titles(question, 10),
    'character_mentions': store.character_mentions(question),
    'returned_anime_metadata': [json.loads(r[0]) for r in catalog.query(
        "SELECT data FROM anime WHERE id IN ('11757','142329','1735','101922','21087','139276')")],
}
Path('data/negative-request-entities.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
