"""Integrity checks on the delivered evaluation data, independent of model output."""
import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from src.recommender.factual_evaluation import FIELDS, load_questions

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/evaluation/hard_facts'


class HardFactsDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (DATA / 'questions.csv').open(encoding='utf-8', newline='') as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.manifest = json.loads((DATA / 'dataset_manifest.json').read_text(encoding='utf-8'))

    def test_size_balance_and_unique_targets(self):
        self.assertEqual(len(self.rows), 180)
        self.assertEqual(len({r['question'].casefold() for r in self.rows}), 180)
        self.assertEqual(len({(r['anime_id'], r['field']) for r in self.rows}), 180)
        counts = Counter(r['field'] for r in self.rows)
        self.assertEqual(set(counts), FIELDS)
        self.assertTrue(all(25 <= n <= 26 for n in counts.values()))

    def test_manifest_matches_delivered_csv(self):
        digest = hashlib.sha256((DATA / 'questions.csv').read_bytes()).hexdigest()
        self.assertEqual(digest, self.manifest['questions_sha256'])
        self.assertEqual(len({r['anime_id'] for r in self.rows}), self.manifest['unique_anime'])
        evidence = [{'question_id': r['question_id'], 'anime_id': r['anime_id'],
                     'field': r['field'], 'values': json.loads(r['expected_values'])} for r in self.rows]
        labels_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(labels_digest, self.manifest['labels_sha256'])

    def test_labels_against_local_database(self):
        database = ROOT / 'data/documents/anilist/catalog.sqlite3'
        if not database.is_file():
            self.skipTest('Local catalog is optional')
        self.assertEqual(len(load_questions(DATA / 'questions.csv', database)), 180)

    def test_authorship_and_tv_duration_scope(self):
        for row in self.rows:
            self.assertEqual(row['question_origin'], 'authored_naturalistic')
            self.assertTrue(json.loads(row['expected_values']))
            self.assertTrue(row['question'].strip())
            self.assertTrue(row['title'].strip())
        # Runtime questions must actually describe episodic entries, not total film runtimes.
        database = ROOT / 'data/documents/anilist/catalog.sqlite3'
        if database.is_file():
            import sqlite3
            with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
                for row in self.rows:
                    if row['field'] == 'duration':
                        data = json.loads(db.execute('SELECT data FROM anime WHERE id=?', (row['anime_id'],)).fetchone()[0])
                        self.assertEqual(data['format'], 'TV')


if __name__ == '__main__':
    unittest.main()
