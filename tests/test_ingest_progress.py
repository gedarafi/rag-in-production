import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config import load_settings
from src.providers import LocalModels
from src.rag import RagPipeline


class IngestProgressTests(unittest.TestCase):
    def test_embedding_batch_order_and_completion(self):
        model = LocalModels.__new__(LocalModels)
        model.embedder = SimpleNamespace(encode=lambda texts, **kwargs: np.array([[int(t), 1] for t in texts], dtype=np.float32))
        events = []
        vectors = model.embed([str(i) for i in range(513)], progress=lambda *event: events.append(event))
        np.testing.assert_array_equal(vectors[:, 0], np.arange(513))
        self.assertEqual([event[1] for event in events], [0, 256, 512, 513])
        self.assertEqual(events[-1], ("Embedding chunks", 513, 513))

    def test_ingest_reports_stages_and_saves_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "documents"
            documents.mkdir()
            (documents / "example.md").write_text("Example knowledge", encoding="utf-8")
            (documents / "empty.md").write_text("", encoding="utf-8")
            (documents / "ignored.json").write_text("{}", encoding="utf-8")
            pipeline = RagPipeline.__new__(RagPipeline)
            pipeline.settings = replace(load_settings(), documents=documents, index=root / "index")
            def embed(texts, progress=None):
                if progress:
                    progress("Embedding chunks", 0, len(texts))
                    progress("Embedding chunks", len(texts), len(texts))
                return np.ones((len(texts), 2))
            pipeline.models = SimpleNamespace(embed=embed)
            events = []
            self.assertEqual(pipeline.ingest(progress=lambda *event: events.append(event)), 1)
            self.assertIn(("Loading documents", 2, 2), events)
            self.assertEqual(events[-1], ("Saving index", 2, 2))
            self.assertEqual(np.load(root / "index/vectors.npy").shape, (1, 2))
            self.assertTrue((root / "index/chunks.json").exists())
