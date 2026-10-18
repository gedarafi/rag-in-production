import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.recommender import vector_db
from src.recommender.chunk_index import write_index
from src.recommender.documents import Chunk
from src.recommender.vector_db import VectorDb


def error(code):
    failure = Exception(code)
    failure.response = {'Error': {'Code': code}}
    return failure


class FakeDataApi:
    """Just enough of the RDS Data API and Postgres for the loader and the search."""

    def __init__(self):
        self.tables, self.indexes, self.statements, self.batches = {}, set(), [], []
        self.resuming = 0  # calls that fail as if the cluster were still resuming
        self.records = []  # what a search returns

    def _wait(self):
        if self.resuming:
            self.resuming -= 1
            raise error('DatabaseResumingException')

    def execute_statement(self, resourceArn, secretArn, database, sql, parameters=(), continueAfterTimeout=False):
        self._wait()
        values = {p['name']: next(iter(p['value'].values())) for p in parameters}
        self.statements.append((sql, values))
        if match := re.match(r'CREATE TABLE ("[^"]+")', sql):
            self.tables[match[1]] = []
        elif match := re.match(r'DROP TABLE IF EXISTS ("[^"]+")', sql):
            self.tables.pop(match[1], None)
        elif match := re.match(r'CREATE INDEX IF NOT EXISTS ("[^"]+")', sql):
            self.indexes.add(match[1])
        elif sql.startswith('SELECT to_regclass'):
            return {'records': [[{'booleanValue': values['name'] in self.tables or values['name'] in self.indexes}]]}
        elif match := re.match(r'SELECT COUNT\(\*\) FROM ("[^"]+")', sql):
            return {'records': [[{'longValue': len(self.tables[match[1]])}]]}
        elif 'FROM pg_tables' in sql:
            return {'records': [[{'stringValue': name.strip('"')}] for name in self.tables]}
        elif sql.startswith(('SELECT', 'WITH')):
            return {'records': self.records}
        return {'records': []}

    def batch_execute_statement(self, resourceArn, secretArn, database, sql, parameterSets):
        self._wait()
        table = re.match(r'INSERT INTO ("[^"]+")', sql)[1]
        self.batches.append(len(json.dumps(parameterSets)))
        self.tables[table].extend({p['name']: next(iter(p['value'].values())) for p in row} for row in parameterSets)
        return {}


class VectorDbTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.api = FakeDataApi()
        self.db = VectorDb('arn:cluster', 'arn:secret', client=self.api, resume_seconds=1, wait_seconds=0)
        self.chunks = [Chunk(f'passage {i} ' + 'ü' * 200, f'anime-{i % 3}.md', str(i % 3), 'anime', None) for i in range(12)]
        self.chunks.append(Chunk('a guide', 'guide.md'))
        vectors = np.random.default_rng(0).normal(size=(len(self.chunks), 4))
        write_index(self.folder, self.chunks, vectors / np.linalg.norm(vectors, axis=1, keepdims=True))

    def test_only_bundle_versions_name_tables(self):
        self.assertEqual(self.db.table('20261018T192902Z').name, '"chunks_20261018t192902z"')
        for version in ('', 'a"; DROP TABLE x; --', 'x' * 41, '../etc'):
            with self.assertRaises(ValueError):
                self.db.table(version)

    def test_load_batches_every_passage_under_the_request_limit_then_indexes(self):
        with patch.object(vector_db, 'BATCH_BYTES', 6000):
            self.assertEqual(self.db.table('v1').load(self.folder), 13)
        rows = self.api.tables['"chunks_v1"']
        self.assertEqual([row['id'] for row in rows], list(range(13)))
        self.assertEqual(sorted(row['text'] for row in rows), sorted(c.text for c in self.chunks))
        self.assertTrue(all(row['character_id'] is True for row in rows))  # isNull
        self.assertGreater(len(self.api.batches), 1)
        self.assertTrue(all(size <= 6000 for size in self.api.batches))
        self.assertEqual(len(json.loads(rows[0]['embedding'])), 4)
        self.assertEqual(self.api.indexes, {'"chunks_v1_anime"', '"chunks_v1_hnsw"'})
        self.assertTrue(self.db.table('v1').ready())
        # Loading again finds everything in place.
        loaded = len(self.api.batches)
        self.db.table('v1').load(self.folder)
        self.assertEqual(len(self.api.batches), loaded)

    def test_a_long_index_build_is_waited_for(self):
        self.db.table('v1').load(self.folder)
        self.api.indexes.clear()
        execute = self.api.execute_statement
        def build_elsewhere(**request):
            if 'USING hnsw' in request['sql']:
                self.api.indexes.add('"chunks_v1_hnsw"')  # the first build, still running when the call fails
                raise error('DatabaseErrorException: duplicate key value violates unique constraint; SQLState: 23505')
            return execute(**request)
        self.api.execute_statement = build_elsewhere
        self.db.table('v1').load(self.folder)
        self.assertTrue(self.db.table('v1').ready())
        self.api.indexes.clear()
        def fail(**request):
            if 'USING hnsw' in request['sql']:
                raise error('DatabaseErrorException: operator class "halfvec_ip_ops" does not exist')
            return execute(**request)
        self.api.execute_statement = fail
        with self.assertRaisesRegex(Exception, 'halfvec_ip_ops'):
            self.db.table('v1').load(self.folder)

    def test_a_resuming_cluster_is_waited_for_and_a_failure_costs_only_the_search(self):
        self.api.resuming = 2
        self.assertEqual(self.db.execute('SELECT 1'), [])
        self.api.resuming = 10 ** 6
        with patch('src.recommender.vector_db.event') as recorded:
            self.assertEqual(self.db.table('v1').search(np.ones(4)), ([], []))
        recorded.assert_called_once()
        with self.assertRaises(Exception):
            self.db.table('v1').load(self.folder)

    def test_search_within_anime_is_exact_and_returns_passages(self):
        self.api.records = [[{'longValue': 3}, {'stringValue': 'passage 3'}, {'stringValue': 'anime-0.md'},
                             {'stringValue': '0'}, {'stringValue': 'anime'}, {'isNull': True}, {'doubleValue': .91}]]
        found, scores = self.db.table('v1').search(np.array([.5, .5, .5, .5]), ['0', 0, '2'], 3)
        self.assertEqual(found, [Chunk('passage 3', 'anime-0.md', '0', 'anime', None)])
        self.assertEqual(scores, [.91])
        sql, values = self.api.statements[-1]
        self.assertIn('MATERIALIZED', sql)
        self.assertEqual((values['ids'], values['k']), ('0,2', 3))
        self.assertEqual(json.loads(values['query']), [.5, .5, .5, .5])
        self.assertEqual(self.db.table('v1').search(np.ones(4), [], 3), ([], []))
        self.db.table('v1').search(np.ones(4), None, 3)
        self.assertNotIn('MATERIALIZED', self.api.statements[-1][0])

    def test_prune_keeps_the_newest_and_the_active_version(self):
        for version in ('20260101T000000Z', '20260201T000000Z', '20260301T000000Z', '20260401T000000Z'):
            self.api.tables[self.db.table(version).name] = []
        self.assertEqual(self.db.prune(2, active='20260101T000000Z'), ['20260201t000000z'])
        self.assertEqual(sorted(self.api.tables), ['"chunks_20260101t000000z"', '"chunks_20260301t000000z"',
                                                   '"chunks_20260401t000000z"'])

    def test_worker_uses_the_bundle_index_else_its_versions_table(self):
        from src.recommender.service import Recommender
        root = self.folder / 'bundle'
        (root / 'data/index').mkdir(parents=True)
        (root / 'recommender.toml').write_text(Path('recommender.toml').read_text())
        (root / 'manifest.json').write_text(json.dumps({'version': 'v7'}))
        environment = {'VECTOR_DB_CLUSTER_ARN': 'arn:cluster', 'VECTOR_DB_SECRET_ARN': 'arn:secret'}
        with patch.dict('os.environ', environment):
            table = Recommender(root).pipeline.passage_table
            self.assertEqual((table.version, table.db.database), ('v7', 'vectors'))
            write_index(root / 'data/index/anilist', self.chunks, np.ones((len(self.chunks), 4)))
            self.assertFalse(hasattr(Recommender(root).pipeline, 'passage_table'))
        (root / 'data/index/anilist/chunks.sqlite3').unlink()
        self.assertFalse(hasattr(Recommender(root).pipeline, 'passage_table'))


if __name__ == '__main__':
    unittest.main()
