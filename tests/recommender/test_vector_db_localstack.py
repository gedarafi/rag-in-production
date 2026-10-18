"""The passage table on the local stack's Aurora in LocalStack (`just test-vector-db`, after `just apply`)."""
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pytest

from src.recommender.chunk_index import ChunkIndex, write_index
from src.recommender.documents import Chunk
from src.recommender.vector_db import VectorDb

pytestmark = pytest.mark.localstack


class LocalStackVectorDbTests(unittest.TestCase):
    def setUp(self):
        self.db = VectorDb.from_env()
        if self.db is None:
            self.skipTest('Needs the local stack: run `just test-vector-db`.')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        rng = np.random.default_rng(7)
        chunks = [Chunk(f'Anime {i % 30}: passage {i}. Ünïcode “quotes” and \'apostrophes\'.', f'anime-{i % 30}.md',
                        str(i % 30), 'character' if i % 2 else 'anime', str(i) if i % 2 else None) for i in range(300)]
        vectors = rng.normal(size=(len(chunks), 384))
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        write_index(self.folder, chunks, self.vectors)
        self.table = self.db.table('test-' + uuid.uuid4().hex[:12])
        self.addCleanup(lambda: self.db.execute(f'DROP TABLE IF EXISTS {self.table.name}'))

    def test_postgres_finds_what_the_file_index_finds(self):
        self.assertEqual(self.table.load(self.folder), 300)
        self.assertTrue(self.table.ready())
        index = ChunkIndex(self.folder)
        for query in (self.vectors[3], self.vectors[150], -self.vectors[42]):
            for anime_ids in (None, ['3', '17'], ['missing']):
                expected, expected_scores = index.search(query, anime_ids, 5)
                found, scores = self.table.search(query, anime_ids, 5)
                self.assertEqual(found, expected)
                np.testing.assert_allclose(scores, expected_scores, atol=1e-3)
