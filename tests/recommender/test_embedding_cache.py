import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.recommender.embedding_cache import cached_embed


class EmbeddingCacheTests(unittest.TestCase):
    def test_reuse_changes_deduplication_and_model_isolation(self):
        calls = []
        def embed(texts):
            calls.extend(texts)
            return np.array([[len(t), 1] for t in texts], dtype=np.float32)
        models = SimpleNamespace(embed=embed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.sqlite3'
            first = cached_embed(['one', 'two', 'one'], models, path, 'model-a')
            self.assertEqual(calls, ['one', 'two'])
            calls.clear()
            np.testing.assert_array_equal(first, cached_embed(['one', 'two', 'one'], models, path, 'model-a'))
            self.assertFalse(calls)
            cached_embed(['one', 'changed'], models, path, 'model-a')
            self.assertEqual(calls, ['changed'])
            calls.clear()
            cached_embed(['one'], models, path, 'model-b')
            self.assertEqual(calls, ['one'])

    def test_resume_after_interrupted_batch(self):
        count = 0
        def fail(texts):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('Interrupted')
            return np.ones((len(texts), 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.sqlite3'
            texts = [str(i) for i in range(300)]
            with self.assertRaisesRegex(RuntimeError, 'Interrupted'):
                cached_embed(texts, SimpleNamespace(embed=fail), path, 'model')
            resumed = []
            def embed(batch):
                resumed.extend(batch)
                return np.ones((len(batch), 2), dtype=np.float32)
            vectors = cached_embed(texts, SimpleNamespace(embed=embed), path, 'model')
            self.assertEqual(resumed, texts[256:])
            self.assertEqual(vectors.shape, (300, 2))
