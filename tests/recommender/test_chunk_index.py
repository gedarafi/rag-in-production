import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.recommender.chunk_index import DATABASE, VECTORS, ChunkIndex, write_index
from src.recommender.documents import Chunk


def normalized(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class ChunkIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        # Chunks of three anime in an interleaved order, plus one with no anime (another knowledge base).
        self.chunks = [Chunk(f'chunk {i} of {anime}', f'anime-{anime}.md', anime, 'anime', None)
                       for i, anime in enumerate(['20', '7', '20', None, '7', '100', '20'])]
        self.chunks[3] = Chunk('guide text', 'guide.md')
        self.vectors = normalized(np.random.default_rng(1).normal(size=(len(self.chunks), 16))).astype(np.float32)
        write_index(self.folder, self.chunks, self.vectors)
        self.index = ChunkIndex(self.folder)

    def brute_force(self, query, keep=lambda chunk: True):
        scores = [(float(v.astype(np.float16).astype(np.float32) @ query), c)
                  for c, v in zip(self.chunks, self.vectors) if keep(c)]
        return [c for _, c in sorted(scores, key=lambda pair: -pair[0])]

    def test_each_anime_is_one_contiguous_slice_in_reading_order(self):
        with closing(sqlite3.connect(self.folder / DATABASE)) as db:
            rows = db.execute('SELECT id, anime_id, text FROM chunks ORDER BY id').fetchall()
            spans = dict((a, (start, stop)) for a, start, stop in db.execute('SELECT * FROM anime_chunks'))
        self.assertEqual(spans, {'100': (1, 2), '20': (2, 5), '7': (5, 7)})
        self.assertEqual([text for _, anime, text in rows if anime == '20'],
                         ['chunk 0 of 20', 'chunk 2 of 20', 'chunk 6 of 20'])
        stored = np.load(self.folder / VECTORS)
        self.assertEqual(stored.dtype, np.float16)
        for row, _, text in rows:
            original = next(i for i, c in enumerate(self.chunks) if c.text == text)
            np.testing.assert_allclose(stored[row], self.vectors[original], atol=1e-3)

    def test_search_matches_brute_force_over_everything_and_within_anime(self):
        query = self.vectors[2]
        found, scores = self.index.search(query, None, 3)
        self.assertEqual(found, self.brute_force(query)[:3])
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0, places=2)
        found, _ = self.index.search(query, ['7', '100'], 10)
        self.assertEqual(found, self.brute_force(query, lambda c: c.anime_id in ('7', '100')))

    def test_filtered_search_reads_only_its_slices_and_k_texts(self):
        queries = []
        query = self.index._query
        with patch.object(self.index, '_query', side_effect=lambda sql, p=(): queries.append(p) or query(sql, p)):
            found, _ = self.index.search(self.vectors[0], ['20'], 2)
        self.assertEqual([c.anime_id for c in found], ['20', '20'])
        self.assertEqual(len(queries[-1]), 2)  # only the two best chunks' texts
        self.assertEqual(self.index.spans(['20', '7']), [(2, 5), (5, 7)])
        self.assertEqual(self.index.spans(['missing']), [])
        self.assertEqual(self.index.search(self.vectors[0], [], 3), ([], []))

    def test_rewrite_replaces_the_earlier_format_and_detects_a_mismatch(self):
        (self.folder / 'vectors.npy').write_bytes(b'old')
        (self.folder / 'chunks.json').write_text('[]')
        write_index(self.folder, self.chunks[:2], self.vectors[:2])
        self.assertFalse((self.folder / 'vectors.npy').exists() or (self.folder / 'chunks.json').exists())
        np.save(self.folder / VECTORS, self.vectors.astype(np.float16))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            ChunkIndex(self.folder)
        with self.assertRaisesRegex(ValueError, 'one vector'):
            write_index(self.folder, self.chunks, self.vectors[:2])


if __name__ == '__main__':
    unittest.main()
