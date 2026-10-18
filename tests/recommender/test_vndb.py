import json
import io
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.recommender.config import load_settings
from src.recommender.vndb import VNDB_URL, decode_copy, plain, prepare_vndb, release_date, rows
from src.recommender.vndb_context import load_context


class VndbTests(unittest.TestCase):
    def test_copy_and_markup(self):
        self.assertIsNone(decode_copy(r"\N"))
        self.assertEqual(decode_copy(r"a\tb\nc\\d\141\x42"), "a\tb\nc\\daB")
        self.assertEqual(decode_copy(r"\\N"), r"\N")
        self.assertEqual(plain("[b]Story[/b] [url=https://example.org]source[/url] [spoiler]secret[/spoiler]"),
                         "Story source (https://example.org) [spoiler omitted]")
        self.assertEqual(release_date("20249999"), "2024")
        self.assertEqual(release_date("20240299"), "2024-02")
        self.assertEqual(release_date("99999999"), "TBA")

    def test_join_and_snapshot_refresh(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "dump"
            (source / "db").mkdir(parents=True)
            (source / "TIMESTAMP").write_text("2026-10-14", encoding="utf-8")

            def table(name, header, content):
                (source / "db" / f"{name}.header").write_text(header + "\n", encoding="utf-8")
                (source / "db" / name).write_text(content, encoding="utf-8")

            table("vn", "id\tolang\tc_votecount\tc_rating\tc_average\tc_length\tc_lengthnum\tlength\tdevstatus\talias\tdescription",
                  "v1\tja\t20\t750\t800\t120\t2\t2\t0\tOther name\tLine one\\nLine two\n")
            table("vn_titles", "id\tlang\tofficial\ttitle\tlatin", "v1\tja\tt\t日本語\tRomanized\nv1\ten\tt\tEnglish title\t\\N\n")
            table("releases", "id\treleased\tofficial\tpatch\tfreeware\tminage", "r1\t20240299\tt\tf\tf\t18\n")
            table("releases_vn", "id\tvid\trtype", "r1\tv1\tcomplete\n")
            table("releases_titles", "id\tlang\tmtl\ttitle\tlatin", "r1\ten\tf\tRelease title\t\\N\n")
            table("releases_platforms", "id\tplatform", "r1\twin\n")
            table("producers", "id\tname\tlatin", "p1\tDeveloper\t\\N\n")
            table("releases_producers", "id\tpid\tdeveloper\tpublisher", "r1\tp1\tt\tf\n")
            table("staff_alias", "id\taid\tname\tlatin", "s1\t1\tWriter\t\\N\n")
            table("vn_staff", "id\taid\trole\teid\tnote", "v1\t1\tscenario\t\\N\t\n")
            table("vn_relations", "id\tvid\trelation\tofficial", "v1\tv2\tseq\tt\n")
            self.assertEqual(prepare_vndb(root, source), 1)
            archive_path = root / "snapshot.tar.zst"
            with tarfile.open(archive_path, "w:zst") as archive:
                archive.add(source, arcname="vndb-snapshot")
            self.assertEqual(prepare_vndb(root, archive_path), 1)
            manifest_path = root / "data/documents/vndb/manifest.json"
            self.assertEqual(json.loads(manifest_path.read_text())["source"], str(archive_path.resolve()))

            def fake_download(url, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive_path, target)
                return target

            with patch("src.recommender.vndb.download", side_effect=fake_download) as download:
                self.assertEqual(prepare_vndb(root), 1)
                download.assert_called_once_with(VNDB_URL, root / "data/raw/vndb-db-latest.tar.zst.part")
                self.assertEqual(prepare_vndb(root), 1)
                self.assertEqual(download.call_count, 1)
            local = root / "data-raw/vndb"
            shutil.copytree(source, local)
            with patch("src.recommender.vndb.download") as download:
                self.assertEqual(prepare_vndb(root), 1)
                download.assert_not_called()
            self.assertEqual(json.loads(manifest_path.read_text())["source"], str(local.resolve()))
            output = root / "data/documents/vndb"
            text = (output / "v1.md").read_text(encoding="utf-8")
            for expected in ("# English title", "日本語", "Line one\nLine two", "2024-02", "Developer", "Writer", "7.50/10", "2.0 hours", "https://vndb.org/r1"):
                self.assertIn(expected, text)
            self.assertNotIn(r"\N", text)
            (output / "v99.md").write_text("old", encoding="utf-8")
            (output / "notes.md").write_text("personal", encoding="utf-8")
            prepare_vndb(root, source)
            self.assertFalse((output / "v99.md").exists())
            self.assertTrue((output / "notes.md").exists())
            reviews = root / "reviews.jsonl"
            reviews.write_text(json.dumps(dict(id="r1", vn_id="v1", author="Reader", date="2026-10-14",
                                               source_url="https://example.org/review", text="Atmospheric", spoiler_level=0)), encoding="utf-8")
            self.assertEqual(prepare_vndb(root, source, reviews=reviews), 2)
            self.assertTrue((output / "review-r1.md").exists())
            catalog = json.loads((output / "catalog.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(catalog["releases"][0]["languages"], ["en"])
            self.assertEqual(catalog["reviews"][0]["author"], "Reader")
            self.assertEqual(prepare_vndb(root, source), 1)
            self.assertFalse((output / "review-r1.md").exists())
            table("vn_titles", "id\tlang", "v1\ten\textra\n")
            with self.assertRaisesRegex(ValueError, "vn_titles:1"):
                list(rows(source, "vn_titles"))

    def test_separate_index(self):
        vndb = load_settings(knowledge_base="vndb")
        self.assertEqual(vndb.documents.name, "vndb")
        self.assertNotEqual(vndb.index, load_settings(knowledge_base="custom").index)

    def test_archive_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "bad.tar.zst"
            with tarfile.open(source, "w:zst") as archive:
                member = tarfile.TarInfo("../escaped.txt")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(tarfile.FilterError):
                prepare_vndb(root, source)

    def test_archive_requires_dump(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "empty.tar.zst"
            with tarfile.open(source, "w:zst"):
                pass
            with self.assertRaisesRegex(ValueError, "exactly one dump"):
                prepare_vndb(root, source)

    def test_enrichment_and_spoiler_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "db").mkdir()

            def table(name, header, content):
                (source / "db" / f"{name}.header").write_text(header + "\n", encoding="utf-8")
                (source / "db" / name).write_text(content, encoding="utf-8")

            table("tags", "id\tname\tcat\talias\tdescription\tdefaultspoil",
                  "g1\tFantasy\tcont\tMagic\tFantasy setting\t0\ng2\tSecret\tcont\t\tHidden setting\t2\n")
            table("tags_vn", "vid\ttag\tvote\tspoiler\tignore\tlie",
                  "v1\tg1\t2\t0\tf\tf\nv1\tg1\t-1\t0\tf\tf\nv1\tg1\t-3\t0\tt\tf\nv1\tg2\t3\t\\N\tf\tf\n")
            table("chars", "id\tdescription\tage", "c1\tA traveler [spoiler]secret identity[/spoiler]\t20\nc2\tSecret character\t21\n")
            table("chars_names", "id\tname\tlatin\tlang", "c1\tAlice\t\\N\ten\nc2\tHidden\t\\N\ten\n")
            table("chars_vns", "id\tvid\trid\trole\tspoil", "c1\tv1\t\\N\tmain\t0\nc2\tv1\t\\N\tside\t2\n")
            table("traits", "id\tname\tdescription\tgid\tdefaultspoil", "i1\tKind\tHelps others\t\\N\t0\ni2\tVillain\tSecret\t\\N\t2\n")
            table("chars_traits", "id\ttid\tspoil\tlie", "c1\ti1\t0\tf\nc1\ti2\t0\tf\n")
            table("anime", "id\ttitle_romaji\ttitle_kanji\ttype\tyear", "1\tAdaptation\t\\N\ttv\t2000\n")
            table("vn_anime", "id\taid", "v1\t1\n")
            reviews = source / "reviews.jsonl"
            review = dict(id="review1", vn_id="v1", author="Reader", date="2026-10-14", text="Strong atmosphere", source_url="https://example.org/review/1", spoiler_level=0)
            reviews.write_text(json.dumps(review) + "\n", encoding="utf-8")
            tags, anime, links, documents, catalog = load_context(source, {"v1": {}}, lambda _: "Example VN", 0, reviews)
            self.assertIn("Fantasy", tags["v1"][0])
            self.assertEqual(catalog["v1"]["tags"][0]["raw_vote_mean"], 0.5)
            self.assertEqual(len(catalog["v1"]["tags"]), 1)
            self.assertIn("Adaptation", anime["v1"][0])
            character = documents["character-c1-v1.md"]
            self.assertIn("Example VN", character)
            self.assertIn("Kind", character)
            self.assertNotIn("Villain", character)
            self.assertNotIn("secret identity", character)
            self.assertNotIn("character-c2-v1.md", documents)
            self.assertIn("Reader opinion", documents["review-review1.md"])
            self.assertEqual(len(links["v1"]), 1)
            _, _, _, full_docs, full_catalog = load_context(source, {"v1": {}}, lambda _: "Example VN", 2, reviews)
            self.assertIn("character-c2-v1.md", full_docs)
            self.assertEqual(len(full_catalog["v1"]["tags"]), 2)
            review["id"] = "../../escape"
            reviews.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid/duplicate"):
                load_context(source, {"v1": {}}, lambda _: "Example VN", 0, reviews)


if __name__ == "__main__":
    unittest.main()
