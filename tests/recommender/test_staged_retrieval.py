import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.recommender.anime_tools import AnimeTools, structured
from src.recommender.config import load_settings
from src.recommender.documents import load_chunks
from src.recommender.providers import LocalModels
from tests.recommender.routing_stub import plan_query
from src.recommender.rag import RagPipeline
from src.recommender.query_router import QueryPlan, fast_answer
from tests.recommender.routing_stub import CatalogRouterStub, echo_evidence


class StagedRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "anilist"
        self.folder.mkdir()
        self.records = [structured({"id": "20", "title_english": "Naruto", "popularity": "900",
            "description": "A young ninja dreams of becoming Hokage.", "episodes": "220", "duration": "23",
            "startDate_year": "2002", "genres": ["Action"], "characters": [{"role": "MAIN", "node": {
                "id": 17, "name": {"full": "Naruto Uzumaki", "native": "うずまきナルト"},
                "description": "A ninja who wants recognition. " + "Training is difficult. " * 80 +
                               "His motivation is to protect his friends and earn acceptance."}}],
            "recommendations": [{"node": {"rating": 40, "mediaRecommendation": {"id": "21"}}}]}, "anime-20.md"),
            structured({"id": "21", "title_english": "One Piece", "popularity": "800", "genres": ["Action"],
                        "description": "Pirates sail the seas."}, "anime-21.md"),
            structured({"id": "934", "title_english": "When They Cry", "title_romaji": "Higurashi no Naku Koro ni",
                        "title_native": "ひぐらしのなく頃に", "popularity": "100", "genres": ["Horror"],
                        "description": "A mystery unfolds in Hinamizawa."}, "anime-934.md")]
        self.write_catalog()
        self.tools = AnimeTools.load(self.folder)
        self.pipeline = RagPipeline(replace(load_settings(), root=self.root, documents=self.folder, index=self.root / "index"))
        self.pipeline._router = CatalogRouterStub()
        self.pipeline._agent = SimpleNamespace(answer=lambda q, h: self.pipeline._ask_retrieval(q, h))
        self.pipeline.models = SimpleNamespace(answer=echo_evidence)

    def write_catalog(self):
        (self.folder / "catalog.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.records), encoding="utf-8")

    def test_sqlite_aliases_relations_and_rankings(self):
        self.assertEqual(self.tools.count(), 3)
        self.assertFalse(hasattr(self.tools, "records"))
        self.assertEqual(self.tools.titles("ひぐらしのなく頃に")[0]["id"], "934")
        records, reasons = self.tools.recommend("I enjoyed Naruto, what should I watch next?")
        self.assertEqual(records[0]["id"], "21")
        self.assertIn("40", reasons["21"])
        self.assertEqual(self.tools.character("Naruto")[0]["id"], 17)
        self.assertEqual(self.tools.popular(1)[0]["id"], "20")
        plan = self.tools.query("EXPLAIN QUERY PLAN SELECT id FROM anime ORDER BY popularity DESC,id LIMIT 1")
        self.assertIn("anime_popularity", str(plan))

    def test_simple_facts_and_overviews_generate_without_vector_search(self):
        questions = {"How many episodes does Naruto have?": "220",
                     "When did Naruto air?": "2002", "What genres is Naruto?": "Action",
                     "Which anime has the largest fanbase?": "Naruto",
                     "Can you tell me who Naruto is?": "Naruto Uzumaki",
                     "Give me a brief summary of Higurashi": "Hinamizawa",
                     "What is When They Cry?": "Hinamizawa",
                     "Who are the main characters in Naruto?": "Naruto Uzumaki"}
        with patch.object(self.pipeline, "vector_search", side_effect=AssertionError("unexpected vectors")):
            for question, expected in questions.items():
                with self.subTest(question=question):
                    self.assertIn(expected, self.pipeline.ask(question))
                    self.assertEqual(self.pipeline.last_retrieval["strategy"], "catalog")
        self.assertFalse(hasattr(self.pipeline.models, 'embed'))

    def test_follow_up_fact_uses_previous_entity(self):
        answer = self.pipeline.ask("How many episodes does it have?", [{"role": "user", "content": "What is Naruto?"}])
        self.assertIn("220", answer)

    def test_typed_recommendation_seed_is_not_confused_with_first_result(self):
        prompts = []
        self.pipeline._router = SimpleNamespace(plan=lambda *args: QueryPlan('recommend', 'Give me anime like that', seed_titles=('Naruto',)))
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *args: prompts.append(prompt) or
            'If you liked Naruto, try One Piece. [anime-21.md]')
        with patch.object(self.pipeline, 'vector_search', side_effect=AssertionError('unexpected large-index retrieval')):
            reply = self.pipeline.ask('Give me anime like that', [{'role':'user', 'content':'What is Naruto about?'}])
        self.assertIn('If you liked Naruto, try One Piece', reply)
        self.assertIn('Seed anime: **Naruto**', prompts[0])
        self.assertIn('1. **One Piece**', prompts[0])

    def test_recommendation_identity_check_accepts_case_and_typographic_variants(self):
        prompts = []
        response = 'If you liked Jujutsu Kaisen, try Hunter x Hunter (2011). [anime-11061.md]'
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *args: prompts.append(prompt) or response)
        evidence = 'Seed anime: **JUJUTSU KAISEN**.\n1. **HUNTER×HUNTER (2011)** — shared tags. [anime-11061.md]'
        answer = self.pipeline.generate_answer('Recommend something similar', evidence, [], catalog=True,
                                               plan=QueryPlan('recommend', 'similar'))
        self.assertEqual(answer, response)
        self.assertEqual(len(prompts), 1)

    def test_duplicate_seed_aliases_do_not_double_recommendation_votes(self):
        records, reasons = self.tools.recommend('Give me anime like that', seed_titles=['Naruto', 'Naruto'])
        self.assertEqual(records[0]['id'], '21')
        self.assertIn('votes: 40.', reasons['21'])

    def test_duration_evidence_includes_episode_count_and_estimated_total(self):
        evidence = fast_answer(self.tools, 'How long is Naruto?', plan=QueryPlan('fact', 'Naruto', field='duration'))
        self.assertIn('220 episodes', evidence)
        self.assertIn('5060 minutes', evidence)

    def test_ollama_routing_uses_json_schema_and_explicit_context_window(self):
        model = LocalModels('unused', 'http://localhost:11434', 'test')
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'response': '{}'})
        with patch('src.recommender.providers.requests.post', return_value=response) as post:
            model.route_plan('prompt')
        request = post.call_args.kwargs['json']
        self.assertIsInstance(request['format'], dict)
        self.assertIn('seed_titles', request['format']['properties'])
        self.assertEqual(request['options']['num_ctx'], 8192)
        self.assertGreaterEqual(request['options']['num_predict'], 512)

    def test_cast_questions_resolve_subject_before_retrieval(self):
        with patch.object(self.pipeline, "vector_search", side_effect=AssertionError("unexpected global search")):
            for question in ("Who is the main character of Naruto?", "Who are the main casts of Naruto?",
                             "Who are the main characters in Naruto?", "Who is Naruto's main character?"):
                self.assertIn("Naruto Uzumaki", self.pipeline.ask(question))
            history = [{'role': 'user', 'content': 'What is Naruto?'}]
            self.assertIn("Naruto Uzumaki", self.pipeline.ask("Who is the main character?", history))
            self.assertEqual(json.loads(self.pipeline.ask("Who is the main character?"))['status'], 'clarification_requested')
            self.assertEqual(json.loads(self.pipeline.ask("Who are the main casts of nonexistent title?", history))['status'], 'not_found')
            self.assertEqual(json.loads(self.pipeline.ask("Who is the main character?",
                             [{'role': 'user', 'content': 'Recommend anime like Naruto'}]))['status'], 'clarification_requested')
        self.assertFalse(hasattr(self.pipeline.models, 'embed'))

    def test_catalog_results_are_evidence_for_natural_answer(self):
        prompts = []
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *args: prompts.append(prompt) or
            'Naruto has 220 episodes. [anime-20.md]')
        with patch.object(self.pipeline, 'vector_search', side_effect=AssertionError('unexpected retrieval')):
            self.assertEqual(self.pipeline.ask('How many episodes does Naruto have?'),
                             'Naruto has 220 episodes.')
        self.assertIn('natural, conversational', prompts[0])
        self.assertIn('220', prompts[0])
        self.assertIn('[anime-20.md]', prompts[0])
        self.assertIn('Do not add facts from memory', prompts[0])

    def test_clarification_is_written_by_the_model(self):
        prompts = []
        reply = 'Which series are you asking about?'
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *args: prompts.append(prompt) or reply)
        self.assertEqual(self.pipeline.ask('Who is the main character?'), reply)
        self.assertIn('clarification_requested', prompts[0])
        self.assertIn('context-specific follow-up', prompts[0])

    def test_missing_title_reply_is_also_written_by_the_model(self):
        prompts = []
        reply = 'I do not see that title in this snapshot. Could you check its spelling?'
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *args: prompts.append(prompt) or reply)
        self.assertEqual(self.pipeline.ask('Who are the main casts of nonexistent title?'), reply)
        self.assertIn('not_found', prompts[0])
        self.assertIn('nonexistent title', prompts[0])

    def test_failed_answer_validation_never_substitutes_template_prose(self):
        self.pipeline.models = SimpleNamespace(answer=lambda *args: 'An unrelated person. [anime-123.md]')
        with self.assertRaises(RuntimeError):
            self.pipeline.ask('Who are the main characters in Naruto?')

    def test_cast_prose_preserves_names_and_rejects_invented_citations(self):
        prompts = []
        def answer_model(prompt, *args):
            prompts.append(prompt)
            return 'Someone else is the protagonist. [anime-123.md]' if len(prompts) == 1 else 'Naruto Uzumaki leads the story. [anime-123.md]'
        self.pipeline.models = SimpleNamespace(answer=answer_model)
        answer = self.pipeline.ask('Who are the main characters in Naruto?')
        self.assertEqual(len(prompts), 2)
        self.assertIn('Naruto Uzumaki', answer)
        self.assertNotIn('[anime-20.md]', answer)
        self.assertNotIn('anime-123.md', answer)
        self.assertNotIn('Someone else', answer)

    def test_details_fetch_relevant_character_passage_without_embeddings(self):
        prompts = []
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *a: prompts.append(prompt) or "Detailed answer")
        with patch.object(self.pipeline, "vector_search", side_effect=AssertionError("unexpected embedding search")):
            answer = self.pipeline.ask("Explain Naruto's motivation in detail")
        self.assertEqual(answer, "Detailed answer")
        self.assertEqual(self.pipeline.last_retrieval["strategy"], "lexical")
        self.assertIn("protect his friends", prompts[0])
        self.assertNotIn("Pirates sail", prompts[0])
        self.assertLess(len(prompts[0]), self.pipeline.settings.context_chars + 1500)

    def test_character_follow_up_retrieves_more_only_when_asked(self):
        self.assertLess(len(self.pipeline.ask("Who is Naruto?")), 1000)
        prompts = []
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *a: prompts.append(prompt) or "More details")
        with patch.object(self.pipeline, "vector_search", side_effect=AssertionError("unexpected vectors")):
            answer = self.pipeline.ask("Tell me more about him", [{"role": "user", "content": "Who is Naruto?"}])
        self.assertEqual(answer, "More details")
        self.assertIn("Naruto Uzumaki", prompts[0])
        self.assertEqual(self.pipeline.last_retrieval["strategy"], "lexical")

    def test_uncatalogued_staff_field_is_opened_only_when_requested(self):
        document = self.folder / "anime-20.md"
        document.write_text('staff: ' + json.dumps([{"role": "Director", "node": {"name": {"full": "Example Director"}}}]), encoding="utf-8")
        prompts = []
        self.pipeline.models = SimpleNamespace(answer=lambda prompt, *a: prompts.append(prompt) or "A director")
        with patch.object(self.pipeline, "vector_search", side_effect=AssertionError("unexpected vectors")):
            self.pipeline.ask("Who directed Naruto?")
        self.assertEqual(self.pipeline.last_retrieval["strategy"], "document")
        self.assertIn("Example Director", prompts[0])

    def test_topic_only_lexical_match_escalates_even_with_known_title(self):
        self.pipeline.models = SimpleNamespace(answer=lambda *a: "Answer")
        with patch.object(self.pipeline, "vector_search", return_value=[]) as semantic:
            self.pipeline.ask("Explain Naruto's quantum biology")
        semantic.assert_called_once()

    def test_unmatched_questions_escalate_and_fuse_lexical_and_vector_evidence(self):
        self.pipeline.settings.index.mkdir()
        np.save(self.pipeline.settings.index / "vectors.npy", np.ones((1, 2)))
        (self.pipeline.settings.index / "chunks.json").write_text(json.dumps([
            {"text": "A young ninja trains to become a leader.", "source": "anime-20.md"}]), encoding="utf-8")
        prompts, embeddings = [], []
        self.pipeline.models = SimpleNamespace(embed=lambda texts: embeddings.append(texts) or np.ones((1, 2)),
            answer=lambda prompt, *a: prompts.append(prompt) or "Answer")
        self.assertEqual(self.pipeline.ask("Explain the difficulties a young ninja faces"), "Answer")
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(self.pipeline.last_retrieval["strategy"], "hybrid")
        self.assertIn("trains to become a leader", prompts[0])

    def test_lexical_metadata_filters_and_chunk_identity(self):
        self.assertFalse(self.tools.lexical("ninja", anime_ids=["934"]))
        passages = self.tools.lexical("motivation", anime_ids=["20"], entity_type="character", character_id=17)
        self.assertTrue(passages)
        self.assertTrue(all(p["character_id"] == "17" and p["anime_id"] == "20" for p in passages))
        for record in self.records:
            (self.folder / record["source"]).write_text("Placeholder", encoding="utf-8")
        chunks = load_chunks(self.folder, (".md",), 300, 30)
        characters = [c for c in chunks if c.entity_type == "character"]
        self.assertTrue(characters)
        self.assertTrue(all(c.anime_id == "20" and c.character_id == "17" for c in characters))
        self.assertTrue(all("Naruto Uzumaki" in c.text for c in characters))

    def test_negative_genre_filter_and_count(self):
        filters = self.tools.filters("anime without horror")
        self.assertEqual(self.tools.count(filters), 2)
        self.assertNotIn("When They Cry", self.tools.route("Recommend anime without horror", plan=plan_query("Recommend anime without horror")))

    def test_snapshot_refresh_is_atomic_and_removes_stale_records(self):
        self.records.pop()
        self.write_catalog()
        refreshed = AnimeTools.load(self.folder)
        self.assertEqual(refreshed.count(), 2)
        self.assertFalse(refreshed.titles("Higurashi"))
        (self.folder / "catalog.jsonl").write_text('{"invalid":', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            AnimeTools.load(self.folder)
        self.assertEqual(refreshed.count(), 2)

    def test_model_generation_does_not_initialize_embedding_model(self):
        model = LocalModels("unused-model", "http://localhost:11434", "test")
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"response": "Answer"})
        with patch("src.recommender.providers.select_device", side_effect=AssertionError("unexpected GPU initialization")), \
             patch("src.recommender.providers.requests.post", return_value=response):
            self.assertEqual(model.answer("prompt", .2, 100), "Answer")
        self.assertIsNone(model._embedder)

    def test_explanations_are_not_mistaken_for_short_synopses(self):
        self.assertEqual(plan_query("What is Naruto's relationship with Sasuke?").intent, "detail")
        self.assertEqual(plan_query("Analyze Higurashi in depth").intent, "detail")


if __name__ == "__main__":
    unittest.main()
