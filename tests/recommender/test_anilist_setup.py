import csv
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.recommender.anilist_setup import AniListWorkspace, STEPS
from src.recommender.config import load_settings
from src.recommender.datasets import ANILIST_FILENAME
from src.recommender.rag import RagPipeline


class AniListSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = replace(load_settings(), root=self.root,
                                documents=self.root / 'data/documents/anilist',
                                index=self.root / 'data/index/anilist')
        self.source = self.root / 'input.csv'
        with self.source.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['id', 'title_english', 'description', 'genres', 'tags', 'characters'])
            writer.writeheader()
            writer.writerow({'id': '1', 'title_english': 'A Small Adventure', 'description': 'A group of friends explores a forest.',
                             'genres': '["Adventure"]',
                             'tags': json.dumps([{'name': 'Friendship', 'description': 'Friends helping one another.'}]),
                             'characters': json.dumps([{'role': 'MAIN', 'node': {'id': 10, 'name': {'full': 'Forest Guide'},
                                                                               'description': 'A guide who helps the travelers.'}}])})
        self.pipeline = RagPipeline(self.settings)
        self.pipeline.models = SimpleNamespace(
            embed=lambda texts, **kwargs: np.ones((len(texts), 2), dtype=np.float32),
            answer=lambda *args: 'READY', cache_revision='test', device='cpu')
        self.workspace = AniListWorkspace(self.settings)

    def run_setup(self, source=None, rebuild=False):
        with patch('src.recommender.anilist_setup.RagPipeline', return_value=self.pipeline):
            return list(self.workspace.setup(source, rebuild))

    def test_fresh_local_setup_builds_sql_all_embeddings_and_ready_manifest(self):
        snapshots = self.run_setup(self.source)
        self.assertTrue(snapshots[-1]['ready'], snapshots[-1]['message'])
        self.assertEqual(len(snapshots), len(STEPS) * 2)
        self.assertTrue(all(s['status'] == 'completed' for s in snapshots[-1]['steps'].values()))
        self.assertTrue(self.workspace.is_prepared())
        self.assertTrue((self.settings.documents / 'catalog.sqlite3').is_file())
        self.assertTrue((self.settings.index / 'embeddings.sqlite3').is_file())
        self.assertTrue((self.settings.index / 'entity-tools.sqlite3').is_file())
        self.assertTrue((self.settings.index / 'entity-features.sqlite3').is_file())
        # One precomputed vector per anime, so recommendations embed only the request.
        record = self.pipeline._agent.catalog.record('1')
        with patch.object(self.pipeline.models, 'embed', side_effect=AssertionError('embedded at request time')):
            self.assertEqual(self.pipeline._agent.anime_vectors([record]).shape, (1, 2))
        self.assertEqual(self.pipeline._agent.feature_candidates('friends')[0]['description'], 'Friends helping one another.')
        self.assertTrue(AniListWorkspace(self.settings).ready)

    def test_second_setup_reuses_index_and_no_download(self):
        self.run_setup(self.source)
        with patch.object(self.pipeline, 'ingest', side_effect=AssertionError('must reuse the index')):
            snapshots = self.run_setup()
        self.assertTrue(snapshots[-1]['ready'], snapshots[-1]['message'])
        self.assertIn('Reusing', snapshots[-1]['steps']['embeddings']['message'])

    def test_failed_rebuild_clears_ready_marker_and_can_retry(self):
        self.run_setup(self.source)
        with patch.object(self.pipeline, 'ingest', side_effect=RuntimeError('embedding interrupted')):
            failed = self.run_setup(rebuild=True)
        self.assertFalse(failed[-1]['ready'])
        self.assertEqual(failed[-1]['steps']['embeddings']['status'], 'error')
        self.assertFalse(self.workspace.manifest.exists())
        self.assertTrue(self.run_setup()[-1]['ready'])

    def test_zip_source_keeps_tag_definitions_and_original_upload(self):
        archive = self.root / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as file:
            file.write(self.source, ANILIST_FILENAME)
        self.assertTrue(self.run_setup(archive)[-1]['ready'])
        self.assertTrue(archive.exists())
        self.assertEqual(self.pipeline._agent.feature_candidates('friends')[0]['description'], 'Friends helping one another.')

    def test_changed_settings_or_index_invalidates_readiness(self):
        self.run_setup(self.source)
        changed = AniListWorkspace(replace(self.settings, embedding='another-model'))
        self.assertFalse(changed.ready)
        (self.settings.index / 'chunk-vectors.npy').write_bytes(b'changed')
        self.assertFalse(self.workspace.is_prepared())
        with self.assertRaisesRegex(RuntimeError, 'recommender build'):
            self.workspace.ask('Hello')

    def test_generation_settings_do_not_invalidate_existing_library(self):
        self.run_setup(self.source)
        # Existing installations may still have these fields in their manifest.
        manifest = self.workspace.signature()
        manifest.update(llm='old-model', ollama_url='http://localhost:11434')
        self.workspace.manifest.write_text(json.dumps(manifest), encoding='utf-8')
        changed = AniListWorkspace(self.settings)
        self.assertTrue(changed.ready)
        changed.pipeline = MagicMock()
        changed.pipeline.ask.return_value = 'An answer'
        self.assertEqual(changed.ask('Hello'), 'An answer')

    def test_missing_marker_still_requires_verification(self):
        self.run_setup(self.source)
        self.workspace.manifest.unlink()
        self.assertFalse(AniListWorkspace(self.settings).ready)

    def test_missing_ready_marker_resumes_without_reembedding(self):
        self.run_setup(self.source)
        self.workspace.manifest.unlink()
        self.workspace = AniListWorkspace(self.settings)
        self.assertIn('Found', self.workspace.steps['embeddings']['message'])
        with patch.object(self.pipeline, 'ingest', side_effect=AssertionError('index already exists')):
            snapshots = self.run_setup()
        self.assertTrue(snapshots[-1]['ready'])
        self.assertIn('Reusing', snapshots[-1]['steps']['embeddings']['message'])

    def test_missing_tools_cache_reuses_document_index(self):
        self.run_setup(self.source)
        (self.settings.index / 'entity-tools.sqlite3').unlink()
        self.workspace = AniListWorkspace(self.settings)
        self.assertFalse(self.workspace.ready)
        with patch.object(self.pipeline, 'ingest', side_effect=AssertionError('only tools are missing')):
            snapshots = self.run_setup()
        self.assertTrue(snapshots[-1]['ready'], snapshots[-1]['message'])
        self.assertTrue((self.settings.index / 'entity-tools.sqlite3').exists())

    def test_failed_final_verification_keeps_embedding_checkpoint_for_retry(self):
        with patch.object(AniListWorkspace, 'verify', side_effect=RuntimeError('verification failed')):
            self.assertFalse(self.run_setup(self.source)[-1]['ready'])
        self.assertTrue((self.settings.index / 'embedding-state.json').exists())
        with patch.object(self.pipeline, 'ingest', side_effect=AssertionError('resume after embeddings')):
            snapshots = self.run_setup()
        self.assertTrue(snapshots[-1]['ready'])

    def test_corrupt_index_is_repaired_instead_of_reused(self):
        self.run_setup(self.source)
        (self.settings.index / 'chunk-vectors.npy').write_bytes(b'incomplete index')
        with patch.object(self.pipeline, 'ingest', wraps=self.pipeline.ingest) as ingest:
            snapshots = self.run_setup()
        ingest.assert_called_once()
        self.assertTrue(snapshots[-1]['ready'], snapshots[-1]['message'])

    def test_legacy_vectors_from_different_model_are_not_reused_even_with_same_dimensions(self):
        self.run_setup(self.source)
        self.workspace.manifest.unlink()
        (self.settings.index / 'embedding-state.json').unlink()
        self.workspace.settings = replace(self.settings, embedding='another-model')
        self.workspace.pipeline = self.pipeline
        self.assertFalse(self.workspace.reusable_index())

    def test_interrupted_download_does_not_leave_a_complete_source(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {'Content-Length': '100'}
        response.iter_content.return_value = iter([b'partial'])
        with patch('src.recommender.anilist_setup.requests.get', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                self.workspace.acquire_dataset(None, False, lambda *args: None)
        self.assertFalse((self.root / 'data/raw' / ANILIST_FILENAME).exists())
        self.assertFalse(list((self.root / 'data/raw').glob('*.part')))

    def test_verification_failure_never_marks_workspace_ready(self):
        with patch.object(AniListWorkspace, 'verify', side_effect=RuntimeError('verification failed')):
            snapshots = self.run_setup(self.source)
        self.assertFalse(snapshots[-1]['ready'])
        self.assertEqual(snapshots[-1]['steps']['verify']['status'], 'error')
        self.assertFalse(self.workspace.manifest.exists())


if __name__ == '__main__':
    unittest.main()
