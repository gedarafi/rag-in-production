import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.config import load_settings
from src.datasets import ANILIST_FILENAME, ANILIST_URL, prepare_anilist, prepare_multidoc2dial


class KnowledgeBaseTests(unittest.TestCase):
    def test_separate_paths(self):
        settings = [load_settings(knowledge_base=name) for name in ("custom", "multidoc2dial", "anilist")]
        self.assertEqual(len({item.documents for item in settings}), 3)
        self.assertEqual(len({item.index for item in settings}), 3)
        self.assertEqual(settings[1].documents, load_settings(knowledge_base="multidoc2dial").documents)
        with self.assertRaises(ValueError):
            load_settings(knowledge_base="unknown")

    def test_csv_and_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "anime.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "title_english", "title_romaji", "description", "genres"])
                writer.writeheader()
                writer.writerow(dict(id="1", title_english="Cowboy Bebop", description="<b>Space</b> &amp; jazz", genres='["Sci-Fi"]'))
                writer.writerow(dict(id="../../2", title_romaji="Japanese title", description="x" * 140000))
            self.assertEqual(prepare_anilist(root, source), 2)
            text = (root / "data/documents/anilist/anime-1.md").read_text(encoding="utf-8")
            self.assertIn("Cowboy Bebop", text)
            self.assertIn("Space  & jazz", text)
            self.assertNotIn("<b>", text)
            archive = root / "anime.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.write(source, "nested/anilist_anime_data_complete.csv")
                file.writestr("fetch_data.py", "raise RuntimeError('must not execute')")
            self.assertEqual(prepare_anilist(root, archive), 2)
            self.assertEqual(len(list((root / "data/documents/anilist").glob("*.md"))), 2)

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "wrong.csv"
            source.write_text("wrong\nvalue\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "title columns"):
                prepare_anilist(Path(folder), source)

    def test_anilist_reuses_local_csv_without_download(self):
        for directory in ("data-raw", "data/raw"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / directory / ANILIST_FILENAME
                source.parent.mkdir(parents=True)
                source.write_text("id,title_english,description\n1,Cowboy Bebop,Space jazz\n", encoding="utf-8")
                with patch("src.datasets.download") as download:
                    self.assertEqual(prepare_anilist(root), 1)
                    download.assert_not_called()
                self.assertIn("Space jazz", (root / "data/documents/anilist/anime-1.md").read_text())

    def test_anilist_download_requests_only_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def single_file_download(url, target):
                target.parent.mkdir(parents=True)
                # Single-file Kaggle responses can still be ZIP containers.
                with zipfile.ZipFile(target, "w") as archive:
                    archive.writestr(ANILIST_FILENAME, "id,title_english\n1,Cowboy Bebop\n")
                return target

            with patch("src.datasets.download", side_effect=single_file_download) as download:
                self.assertEqual(prepare_anilist(root), 1)
                download.assert_called_once_with(ANILIST_URL, root / "data/raw" / ANILIST_FILENAME)
                self.assertTrue(ANILIST_URL.endswith("/" + ANILIST_FILENAME))
                self.assertEqual(prepare_anilist(root), 1)
                self.assertEqual(download.call_count, 1)

    def test_multidoc_source_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "docs.json"
            source.write_text('{"doc_data":{"va":{"benefits":{"title":"Benefits","doc_text":"Education benefits"}}}}', encoding="utf-8")
            self.assertEqual(prepare_multidoc2dial(root, source), 1)
            self.assertEqual(prepare_multidoc2dial(root), 1)
            self.assertIn("Education benefits", (root / "data/documents/multidoc2dial/va/benefits.md").read_text())


if __name__ == "__main__":
    unittest.main()
