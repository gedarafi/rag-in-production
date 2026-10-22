import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import boto3
from moto import mock_aws

from src.recommender import bundle
from src.recommender.config import load_settings


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / 'workspace'
        self.settings = replace(load_settings(), root=root, documents=root / 'data/documents/anilist',
                                index=root / 'data/index/anilist')
        self.config = root / 'settings.toml'
        for folder in (self.settings.documents, self.settings.index, root / 'prompts'):
            folder.mkdir(parents=True)
        self.config.write_text(Path('settings.toml').read_text(encoding='utf-8'), encoding='utf-8')
        (root / 'prompts/answer.txt').write_text('Answer.', encoding='utf-8')
        with closing(sqlite3.connect(self.settings.documents / 'catalog.sqlite3')) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE anime(id)')
            db.execute('INSERT INTO anime VALUES (1)')
            db.commit()
        (self.settings.documents / 'catalog.jsonl').write_text('{}\n', encoding='utf-8')
        (self.settings.documents / 'anime-1.md').write_text('title: One', encoding='utf-8')
        (self.settings.index / 'vectors.npy').write_bytes(b'vectors')
        (self.settings.index / 'chunks.json').write_text('[]', encoding='utf-8')
        (self.settings.index / 'embeddings.sqlite3').write_bytes(b'build cache')
        (self.settings.index / 'setup.json').write_text('{}', encoding='utf-8')
        self.target = Path(self.temp.name) / 'server'
        mock = mock_aws()
        mock.start()
        self.addCleanup(mock.stop)
        self.s3 = boto3.client('s3', region_name='us-east-1')
        self.s3.create_bucket(Bucket='data')

    def publish(self):
        return bundle.publish(self.settings, self.config, 'data', s3=self.s3)

    def test_round_trip_serves_runtime_files_only(self):
        manifest = self.publish()
        release = bundle.pull('data', self.target, s3=self.s3)
        self.assertEqual(release.name, manifest['version'])
        self.assertEqual((self.target / 'current').resolve(), release.resolve())
        self.assertTrue((release / 'settings.toml').is_file())
        self.assertTrue((release / 'prompts/answer.txt').is_file())
        self.assertTrue((release / 'data/documents/anilist/anime-1.md').is_file())
        for left_out in ('data/documents/anilist/catalog.jsonl', 'data/index/anilist/embeddings.sqlite3',
                         'data/index/anilist/setup.json'):
            self.assertFalse((release / left_out).exists(), left_out)
        # The WAL was folded into the database before packing.
        with closing(sqlite3.connect(release / 'data/documents/anilist/catalog.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT id FROM anime').fetchall(), [(1,)])
        settings = load_settings(release / 'settings.toml', 'anilist')
        self.assertEqual(settings.documents, (release / 'data/documents/anilist').resolve())

    def test_a_prompt_override_folder_ships_with_its_settings(self):
        root = self.settings.root
        overrides = root / 'prompts/overrides'
        overrides.mkdir()
        (overrides / 'agent_reply_system.j2').write_text('Answer briefly.\n', encoding='utf-8')
        self.settings = replace(self.settings, prompt_overrides=overrides)
        self.config.write_text(self.config.read_text(encoding='utf-8').replace(
            '# prompt_overrides = ""', 'prompt_overrides = "prompts/overrides"'), encoding='utf-8')
        self.publish()
        release = bundle.pull('data', self.target, s3=self.s3)
        from src.recommender.prompt_library import prompts_for
        settings = load_settings(release / 'settings.toml', 'anilist')
        self.assertEqual(prompts_for(settings).render('agent_reply_system'), 'Answer briefly.')

    def test_pulling_the_current_version_again_downloads_nothing(self):
        self.publish()
        bundle.pull('data', self.target, s3=self.s3)
        self.s3.delete_object(Bucket='data', Key=self.s3.list_objects_v2(Bucket='data')['Contents'][0]['Key'])
        self.assertTrue(bundle.pull('data', self.target, s3=self.s3).is_dir())

    def test_checksum_mismatch_is_never_served(self):
        manifest = self.publish()
        key = f"recommender/{manifest['version']}/manifest.json"
        self.s3.put_object(Bucket='data', Key=key, Body=json.dumps({**manifest, 'bundle_sha256': '0' * 64}).encode())
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            bundle.pull('data', self.target, s3=self.s3)
        self.assertFalse((self.target / 'current').exists())

    def test_unbuilt_library_is_not_published(self):
        (self.settings.index / 'vectors.npy').unlink()
        with self.assertRaisesRegex(RuntimeError, 'recommender-setup'):
            self.publish()


if __name__ == '__main__':
    unittest.main()
