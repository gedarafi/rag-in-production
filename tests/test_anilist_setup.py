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
import requests

from src.anilist_setup import AniListWorkspace, STEPS
from src.config import load_settings
from src.datasets import ANILIST_FILENAME
from src.rag import RagPipeline


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
        self.tags = MagicMock()
        self.tags.json.return_value = {'models': [{'name': self.settings.llm}]}

    def run_setup(self, source=None, rebuild=False):
        with patch('src.anilist_setup.RagPipeline', return_value=self.pipeline), \
             patch('src.anilist_setup.requests.get', return_value=self.tags):
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
        (self.settings.index / 'chunks.json').write_text('[]', encoding='utf-8')
        self.assertFalse(self.workspace.is_prepared())
        with self.assertRaisesRegex(RuntimeError, 'Run setup'):
            self.workspace.ask('Hello')

    def test_ollama_unavailable_reports_retry_and_keeps_chat_disabled(self):
        with patch('src.anilist_setup.requests.get', side_effect=requests.ConnectionError('offline')):
            snapshots = list(self.workspace.setup(self.source))
        self.assertFalse(snapshots[-1]['ready'])
        self.assertIn('ollama serve', snapshots[-1]['message'])
        self.assertFalse(self.workspace.lock.locked())

    def test_interrupted_download_does_not_leave_a_complete_source(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {'Content-Length': '100'}
        response.iter_content.return_value = iter([b'partial'])
        with patch('src.anilist_setup.requests.get', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                self.workspace.acquire_dataset(None, False, lambda *args: None)
        self.assertFalse((self.root / 'data/raw' / ANILIST_FILENAME).exists())
        self.assertFalse(list((self.root / 'data/raw').glob('*.part')))

    def test_missing_generation_model_is_pulled_with_progress(self):
        self.tags.json.return_value = {'models': []}
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [b'{"status":"downloading","completed":2,"total":4}', b'{"status":"success"}']
        progress = []
        with patch('src.anilist_setup.requests.get', return_value=self.tags), \
             patch('src.anilist_setup.requests.post', return_value=response) as pull:
            self.workspace.ensure_model(lambda *event: progress.append(event))
        self.assertEqual(progress[0], ('Ollama: downloading', 2, 4))
        self.assertTrue(pull.call_args.kwargs['stream'])

    def test_generation_failure_never_marks_workspace_ready(self):
        self.pipeline.models.answer = lambda *args: ''
        snapshots = self.run_setup(self.source)
        self.assertFalse(snapshots[-1]['ready'])
        self.assertEqual(snapshots[-1]['steps']['verify']['status'], 'error')
        self.assertFalse(self.workspace.manifest.exists())


if __name__ == '__main__':
    unittest.main()
