import csv
import json
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from src.anime_tools import AnimeTools, structured
from src.config import load_settings
from src.feature_dictionary import feature_definitions


class FeatureDictionaryTests(unittest.TestCase):
    def test_imported_tag_descriptions_are_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            documents = root / 'data/documents/anilist'
            documents.mkdir(parents=True)
            raw = root / 'data-raw'
            raw.mkdir()
            tags = [{'name': 'Tragedy', 'description': 'Tragic events and unhappy endings.', 'category': 'Theme-Drama'}]
            with (raw / 'anilist_anime_data_complete.csv').open('w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['id', 'tags'])
                writer.writeheader()
                writer.writerow({'id': '1', 'tags': json.dumps(tags)})
            catalog = SimpleNamespace(database=documents / 'catalog.sqlite3')
            with closing(sqlite3.connect(catalog.database)) as db:
                db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
                db.commit()
            settings = replace(load_settings(), root=root, documents=documents)
            first = feature_definitions(settings, catalog)
            self.assertEqual(first['tragedy']['description'], 'Tragic events and unhappy endings.')
            self.assertEqual(feature_definitions(settings, catalog), first)

    def test_recommendation_tags_are_retained_even_when_spoiler_marked(self):
        row = {'id': '1', 'title_english': 'A Drama',
               'tags': [{'name': 'Tragedy', 'isGeneralSpoiler': True, 'isMediaSpoiler': True}]}
        self.assertIn('Tragedy', structured(row, 'anime-1.md')['tags'])

    def test_old_catalog_restores_missing_filter_tags_from_original_document(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            tags = [{'name': 'Tragedy', 'isGeneralSpoiler': True}]
            record = structured({'id': '1', 'title_english': 'A Drama', 'popularity': 100, 'tags': tags}, 'anime-1.md')
            record.pop('tag_schema')
            record['tags'] = []
            (folder / 'catalog.jsonl').write_text(json.dumps(record), encoding='utf-8')
            (folder / 'anime-1.md').write_text('tags: ' + json.dumps(tags) + '\n', encoding='utf-8')
            catalog = AnimeTools.load(folder)
            self.assertIn('Tragedy', catalog.record('1')['tags'])
            self.assertEqual(len(catalog.popular(5, {'terms': ['Tragedy']})), 1)
