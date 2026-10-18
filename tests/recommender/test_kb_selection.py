import tempfile
import unittest
from pathlib import Path

from src.recommender.config import load_settings, select_knowledge_base


class SelectionTests(unittest.TestCase):
    def test_persist_and_preserve_other_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "recommender.toml"
            original = Path("recommender.toml").read_text(encoding="utf-8")
            config.write_text(original, encoding="utf-8")
            before = load_settings(config)
            select_knowledge_base(config, "vndb")
            after = load_settings(config)
            self.assertEqual(after.documents, config.parent / "data/documents/vndb")
            self.assertEqual(after.index, config.parent / "data/index/vndb")
            self.assertEqual(before.embedding, after.embedding)
            self.assertEqual(before.chunk_size, after.chunk_size)
            selected_text = config.read_text(encoding="utf-8")
            self.assertIn('default = "vndb"', selected_text)
            self.assertIn("# Settings of the anime recommender", selected_text)
            with self.assertRaises(ValueError):
                select_knowledge_base(config, "invalid")
            self.assertEqual(config.read_text(encoding="utf-8"), selected_text)

    def test_config_without_selection_section(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "recommender.toml"
            import re
            original = Path("recommender.toml").read_text(encoding="utf-8")
            config.write_text(re.sub(r"(?ms)^\[knowledge_base\].*?(?=^\[)", "", original), encoding="utf-8")
            select_knowledge_base(config, "anilist")
            self.assertEqual(load_settings(config).documents.name, "anilist")
