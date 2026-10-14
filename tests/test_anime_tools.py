import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.anime_tools import AnimeTools, normalize, structured
from src.config import load_settings
from src.datasets import prepare_anilist
from src.documents import load_chunks
from src.rag import RagPipeline
from tests.routing_stub import CatalogRouterStub, echo_evidence


def anime(identifier, title, popularity, genres, **extra):
    return structured({"id": identifier, "title_english": title, "popularity": popularity,
                       "genres": genres, **extra}, f"anime-{identifier}.md")


class AnimeToolTests(unittest.TestCase):
    def route(self, question, history=None):
        plan = CatalogRouterStub().plan(question, history or [], self.tools)
        return self.tools.route(question, history, plan=plan)

    def setUp(self):
        self.records = [
            anime(1, "When They Cry", 100, ["Horror", "Mystery"],
                  title_romaji="Higurashi no Naku Koro ni", title_native="ひぐらしのなく頃に",
                  description="A mystery unfolds in Hinamizawa.",
                  tags=[{"name": "Time Loop"}],
                  recommendations=[{"node": {"rating": 30, "mediaRecommendation": {"id": 3}}}]),
            anime(2, "Naruto", 900, ["Action"], characters=[{"role": "MAIN", "name": None,
                  "node": {"id": 17, "name": {"full": "Naruto Uzumaki", "native": "うずまきナルト"},
                           "description": "__Height:__ 145 cm\n\nA determined ninja who dreams of becoming Hokage. ~!Secret parent.!~"}}]),
            anime(3, "Another", 300, ["Horror", "Mystery"], tags=[{"name": "Time Loop"}]),
            anime(4, "Space Show", 5000, ["Sci-Fi"], startDate_year="2020"),
            anime(5, "Missing Popularity", "nan", ["Horror"]),
            anime(6, "Unrelated Show", 1, ["Comedy"], characters=[{"role": "MAIN", "node": {
                  "name": {"full": "Naruto"}, "description": "An unrelated character."}}]),
            anime(7, "with", 1, ["Comedy"]),
        ]
        self.tools = AnimeTools(self.records)

    def test_numeric_popularity_scans_whole_catalog(self):
        self.assertEqual([r["id"] for r in self.tools.popular(3)], ["4", "2", "3"])
        self.assertIn("5,000", self.route("What's the most popular anime?"))
        self.assertIn("anime-4.md", self.route("What's the most popular anime?"))

    def test_genre_tag_and_year_filters(self):
        result, _ = self.tools.recommend("Recommend Horror Mystery anime with Time Loop")
        self.assertEqual({r["id"] for r in result}, {"1", "3"})
        result = self.tools.popular(5, self.tools.filters("most popular Sci-Fi anime from 2020"))
        self.assertEqual([r["id"] for r in result], ["4"])
        self.assertNotIn("Space Show", self.route("most popular horror anime"))

    def test_recommendation_by_partial_romaji_and_excludes_seed(self):
        records, reasons = self.tools.recommend("Recommend anime similar to Higurashi")
        self.assertEqual(records[0]["id"], "3")
        self.assertNotIn("1", [r["id"] for r in records])
        self.assertIn("30", reasons["3"])
        self.assertIn("Time Loop", reasons["3"])

    def test_unknown_recommendation_does_not_return_random_popular_titles(self):
        self.assertEqual(self.tools.recommend("recommend anime like nonexistent title")[0], [])

    def test_japanese_romaji_and_english_aliases(self):
        for name in ("Higurashi", "ひぐらしのなく頃に", "When They Cry"):
            answer = self.route(f"What is {name}?")
            self.assertIn("Hinamizawa", answer)
            self.assertIn("anime-1.md", answer)
        self.assertEqual(normalize("Ｓｔｅｉｎｓ；Ｇａｔｅ"), normalize("Steins;Gate"))
        self.assertNotEqual(normalize("ガ"), normalize("カ"))

    def test_character_nested_node_description_not_just_name(self):
        for name in ("Naruto", "Naruto Uzumaki", "うずまきナルト"):
            answer = self.route("Who is " + name + "?")
            self.assertIn("dreams of becoming Hokage", answer)
            self.assertNotIn("Secret parent", answer)
            self.assertIn("anime-2.md", answer)
        self.assertIn("dreams of becoming Hokage", self.route("Please describe Naruto from Naruto"))

    def test_missing_character_description_is_explicit(self):
        self.records[1]["characters"][0]["description"] = ""
        self.assertEqual(json.loads(self.route("Who is Naruto?"))['status'], 'missing_field')

    def test_follow_up_recommendation_uses_previous_subject(self):
        answer = self.route("Recommend something similar", [{"role": "user", "content": "What is Higurashi?"}])
        self.assertIn("Another", answer)

    def test_catalog_answers_generate_without_a_vector_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "anilist"
            documents.mkdir()
            path = documents / "catalog.jsonl"
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.records), encoding="utf-8")
            pipeline = RagPipeline(replace(load_settings(), root=root, documents=documents, index=root / "index"))
            pipeline._router = CatalogRouterStub()
            pipeline._agent = SimpleNamespace(answer=lambda q, h: pipeline._ask_retrieval(q, h))
            pipeline.models = SimpleNamespace(answer=echo_evidence)
            self.assertIn("Space Show", pipeline.ask("What is most popular?"))
            self.assertIn("Hokage", pipeline.ask("Who is Naruto?"))
            self.assertFalse(hasattr(pipeline.models, 'embed'))
            self.assertFalse(pipeline.settings.index.exists())
            self.records[3]["popularity"] = 0
            path.write_text("\n".join(json.dumps(r) for r in self.records), encoding="utf-8")
            self.assertEqual(pipeline.anime_tools().popular(1)[0]["title"], "Naruto")

    def test_import_preserves_json_and_semantic_chunks_have_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "anime.csv"
            character = {"role": "MAIN", "node": {"id": 17, "name": {"full": "Naruto Uzumaki"},
                         "description": "A <b>ninja</b> called &quot;hero&quot;. " + "Determined. " * 100}}
            row = {"id": "2", "title_english": "Naruto", "title_native": "ナルト", "characters": json.dumps([character])}
            with source.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            prepare_anilist(root, source)
            documents = root / "data/documents/anilist"
            exported = (documents / "anime-2.md").read_text(encoding="utf-8")
            characters = json.loads(next(line.split(": ", 1)[1] for line in exported.splitlines() if line.startswith("characters:")))
            self.assertEqual(characters[0]["node"]["description"], character["node"]["description"])
            chunks = load_chunks(documents, (".md",), 200, 20)
            self.assertTrue(all("Anime: Naruto" in c.text for c in chunks))
            self.assertFalse(any("voiceActors" in c.text for c in chunks))
            self.assertTrue(any('"hero"' in c.text for c in chunks))


if __name__ == "__main__":
    unittest.main()
