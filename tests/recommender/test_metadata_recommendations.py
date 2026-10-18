import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.recommender.anime_tools import AnimeTools, structured
from src.recommender.config import load_settings
from src.recommender.entity_agent import EntityAgent
from src.recommender.entity_store import EntityStore
from src.recommender.tool_plan import plan_schema, validate_plan


class MetadataRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.records = []
        for aid, year, score, studio, material, role in [
            ('1', 1989, 96, 'Studio One', 'MANGA', 'Director'),
            ('2', 1990, 80, 'Studio One', 'MANGA', 'Director'),
            ('3', 1995, 95, 'Studio Two', 'ORIGINAL', 'Music'),
            ('4', 1999, 90, 'Studio One', 'LIGHT_NOVEL', 'Director'),
            ('5', 2000, 85, 'Studio Two', 'MANGA', 'Director'),
            ('6', None, 99, 'Studio One', 'MANGA', 'Director')]:
            self.records.append(structured({
                'id': aid, 'title_english': 'Show ' + aid, 'startDate_year': year,
                'averageScore': score, 'popularity': 1000 - int(aid), 'format': 'TV',
                'description': 'A documented story.', 'genres': ['Drama'], 'tags': [], 'source': material,
                'episodes': [None, 12, 24, 6, 50, 1][int(aid) - 1], 'duration': 24,
                'status': 'FINISHED', 'season': 'FALL', 'countryOfOrigin': 'JP', 'isLicensed': True,
                'relations': [{'relationType': 'PREQUEL', 'node': {'title': {'english': 'Earlier Story'}}}],
                'characters': [{'role': 'MAIN', 'node': {'id': int(aid), 'name': {'full': 'Hero ' + aid},
                                 'description': 'A detective.'},
                                'voiceActors': [{'name': {'full': 'Example Actor'}, 'languageV2': 'Japanese'}]}],
                'studios': [{'isMain': True, 'node': {'name': studio, 'isAnimationStudio': True}}],
                'staff': [{'role': role, 'node': {'name': {'full': 'Alice Example'}}},
                          {'role': 'Director', 'node': {'name': {'full': 'Bob Person'}}}],
            }, 'anime-' + aid + '.md'))
        (self.folder / 'catalog.jsonl').write_text('\n'.join(json.dumps(r) for r in self.records), encoding='utf-8')
        self.catalog = AnimeTools.load(self.folder)
        self.store = EntityStore(self.catalog)

    def recommend(self, **constraints):
        return self.store.recommendations([], [], 20, constraints)['items']

    def test_arbitrary_inclusive_range_and_score_sort(self):
        items = self.recommend(year_from=1990, year_to=1999, sort_by='score')
        self.assertEqual([r['id'] for r in items], ['3', '4', '2'])
        items = self.recommend(year_from=1989, year_to=1995, sort_by='score')
        self.assertEqual([r['id'] for r in items], ['1', '3', '2'])

    def test_preferred_count_is_ranked_before_limit_without_excluding_other_lengths(self):
        constraints = {'filters': [{'field': 'episodes', 'operator': 'lte', 'value': '50'}],
                       'numeric_preferences': [{'field': 'episodes', 'min': 10, 'max': 19}]}
        self.assertEqual(self.store.recommendations([], [], 1, constraints)['items'][0]['episodes'], 12)
        items = self.store.recommendations([], [], 20, constraints)['items']
        self.assertEqual(items[0]['episodes'], 12)
        self.assertIn(50, [r['episodes'] for r in items])
        self.assertNotIn(None, [r['episodes'] for r in items])
        self.assertEqual(self.catalog.popular(1, constraints)[0]['episodes'], 12)

    def test_general_field_bounds_and_sorting_exclude_missing_counts(self):
        filters = [{'field': 'episodes', 'operator': 'gte', 'value': '6'},
                   {'field': 'episodes', 'operator': 'lte', 'value': '12'},
                   {'field': 'status', 'operator': 'eq', 'value': 'finished'}]
        items = self.recommend(filters=filters, sort_by='episodes', sort_order='asc')
        self.assertEqual([r['id'] for r in items], ['4', '2'])
        self.assertEqual(items[1]['episodes'], 12)
        self.assertEqual(items[1]['matched_fields']['episodes'], 12)
        seeded = self.store.recommendations(['3'], [], 10,
            {'filters': filters, 'sort_by': 'episodes', 'sort_order': 'asc'})
        self.assertEqual([r['id'] for r in seeded['items']], ['4', '2'])

    def test_export_fields_voice_actors_and_character_data_are_searchable(self):
        for field, op, value in [('countryOfOrigin', 'eq', 'jp'), ('isLicensed', 'eq', 'true'),
                                 ('season', 'eq', 'fall'), ('voice_actors', 'contains', 'Example Actor'),
                                 ('characters', 'contains', 'detective'), ('relations', 'contains', 'Earlier Story')]:
            with self.subTest(field=field):
                self.assertEqual(len(self.recommend(filters=[{'field': field, 'operator': op, 'value': value}])), 6)
        self.assertEqual(self.catalog.searchable_fields()['voice_actors'], 'collection')
        self.assertEqual(self.recommend(filters=[{'field': 'voice_actors', 'operator': 'contains', 'value': 'Nobody'}]), [])

    def test_field_filter_validation_never_silently_discards_constraints(self):
        with self.assertRaisesRegex(ValueError, 'field filters'):
            self.recommend(filters={})
        for item in [dict(field='not_real', operator='eq', value='x'),
                     dict(field='episodes', operator='lte', value="12 OR 1=1"),
                     dict(field='title', operator='gt', value='1'),
                     dict(field='isLicensed', operator='eq', value='maybe')]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.recommend(filters=[item])
        self.assertEqual(self.recommend(filters=[dict(field='episodes', operator='lte', value='0')]), [])

    def test_field_filter_schema_uses_string_values(self):
        plan = {'scope': 'in_domain', 'direct_response': '', 'use_previous_context': False,
                'goal': 'Choose short candidates.', 'clarification': '', 'steps': [{
                    'tool': 'recommend_anime', 'arguments': {'filters': [
                        {'field': 'episodes', 'operator': 'lte', 'value': '13'}]}}]}
        self.assertEqual(validate_plan(json.dumps(plan), plan_schema(8)), plan)
        plan['steps'][0]['arguments']['filters'][0]['value'] = 13
        with self.assertRaisesRegex(ValueError, 'field filter'):
            validate_plan(json.dumps(plan), plan_schema(8))
        plan['steps'][0]['arguments']['filters'][0] = {'field': 'invented', 'operator': 'eq', 'value': 'yes'}
        with self.assertRaisesRegex(ValueError, 'Unknown search field'):
            validate_plan(json.dumps(plan), plan_schema(8, self.catalog.searchable_fields()))

    def test_large_matching_fields_keep_relevant_values_in_the_answer_window(self):
        from src.recommender.catalog_search import evidence_value
        value = [{'name': 'Other ' + str(i), 'biography': 'Long unrelated description. ' * 100} for i in range(30)]
        value.append({'name': 'Example Actor', 'language': 'Japanese'})
        result = evidence_value(value, 'Example Actor')
        self.assertLess(len(json.dumps(result)), 1400)
        self.assertIn('Example Actor', json.dumps(result))

    def test_short_and_beginner_requests_receive_grounded_candidates_without_extra_calls(self):
        for question, arguments in [('Recommend me a short anime', {'filters': [
                {'field': 'episodes', 'operator': 'lte', 'value': '13'}], 'sort_by': 'score'}),
                ('Recommend me an anime for beginners', {'query': 'accessible introductory anime', 'sort_by': 'popularity'})]:
            with self.subTest(question=question):
                plan = {'scope': 'in_domain', 'direct_response': '', 'use_previous_context': False,
                        'goal': 'Recommend suitable candidates with stated assumptions.', 'clarification': '', 'steps': [
                            {'tool': 'recommend_anime', 'arguments': arguments}]}
                prompts = []
                def reply(prompt, *_):
                    prompts.append(prompt)
                    return 'Show 4 and Show 2 are possible starting choices.'
                models = SimpleNamespace(embed=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
                                         cache_revision='test', plan_tools=lambda *_: json.dumps(plan), agent_reply=reply)
                agent = EntityAgent(replace(load_settings(), documents=self.folder, index=self.folder / 'index'), models, self.catalog)
                self.assertIn('Show 4', agent.answer(question, []))
                self.assertEqual(agent.last_usage['llm_calls'], 2)
                self.assertTrue(agent.last_trace[0]['result']['items'])
                self.assertIn('recommendation judgment', prompts[0])

    def test_open_ended_timeframes_exclude_unknown_dates(self):
        self.assertEqual({r['id'] for r in self.recommend(year_to=1990)}, {'1', '2'})
        self.assertEqual({r['id'] for r in self.recommend(year_from=1999)}, {'4', '5'})

    def test_seed_similarity_can_combine_with_timeframe_and_score_ranking(self):
        result = self.store.recommendations(['3'], [], 5,
            {'year_from': 1990, 'year_to': 2000, 'sort_by': 'score'})
        self.assertEqual([r['id'] for r in result['items']], ['4', '5', '2'])

    def test_combined_studio_material_and_timeframe(self):
        items = self.recommend(year_from=1990, year_to=1999, studio='studio one', source_material='light novel')
        self.assertEqual([r['id'] for r in items], ['4'])
        self.assertEqual(items[0]['source_material'], 'LIGHT_NOVEL')
        self.assertEqual(items[0]['studios'][0]['name'], 'Studio One')

    def test_staff_and_role_must_match_the_same_person(self):
        items = self.recommend(year_from=1990, year_to=1999, staff='Alice Example', staff_role='Director')
        self.assertEqual({r['id'] for r in items}, {'2', '4'})
        self.assertEqual(items[0]['matched_staff'], [{'name': 'Alice Example', 'role': 'Director'}])
        music = self.recommend(staff='Example Alice', staff_role='Music')
        self.assertEqual([r['id'] for r in music], ['3'])

    def test_unmatched_metadata_does_not_widen_results(self):
        self.assertEqual(self.recommend(studio='Unknown Studio'), [])
        self.assertEqual(self.recommend(source_material='GAME'), [])

    def test_alternative_tags_are_or_candidates_but_required_filters_stay_and(self):
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.catalog.database)) as db:
            db.executemany('INSERT INTO features VALUES (?,?,?)',
                           [('2', 'anti hero', 'Anti-Hero'), ('3', 'revenge', 'Revenge'),
                            ('5', 'revenge', 'Revenge')])
            db.commit()
        rows = self.catalog.popular(10, {'terms': ['Drama'], 'any_terms': ['Anti-Hero', 'Revenge'],
                                        'year_from': 1990, 'year_to': 1999})
        self.assertEqual({r['id'] for r in rows}, {'2', '3'})
        self.assertEqual(self.catalog.popular(10, {'terms': ['Anti-Hero', 'Revenge']}), [])

    def test_specific_work_metadata_is_available_in_grounded_sections(self):
        overview = self.store.anime('2')
        self.assertEqual(overview['source_material'], 'MANGA')
        self.assertEqual(overview['source'], 'anime-2.md')
        staff = self.store.anime('2', section='staff', limit=1)
        self.assertEqual(staff['staff'], [{'name': 'Alice Example', 'role': 'Director'}])
        self.assertEqual(staff['next_offset'], 1)
        self.assertEqual(self.store.anime('2', section='staff', offset=1)['staff'][0]['name'], 'Bob Person')
        self.assertEqual(self.store.anime('2', section='studios')['studios'][0]['name'], 'Studio One')

    def test_recommendation_without_seed_or_genres_and_invalid_range(self):
        agent = EntityAgent(load_settings(), SimpleNamespace(), self.catalog)
        state = {'known': {k: {} for k in ('anime', 'character', 'genre', 'tag')}}
        result = agent.execute('recommend_anime', {'year_from': 1990, 'year_to': 1999, 'sort_by': 'score'}, state)
        self.assertEqual([r['id'] for r in result['items']], ['3', '4', '2'])
        result = agent.execute('recommend_anime', {'limit': 2}, state)
        self.assertEqual(len(result['items']), 2)
        with self.assertRaisesRegex(ValueError, 'timeframe'):
            agent.execute('recommend_anime', {'year_from': 2000, 'year_to': 1990}, state)

    def test_timeframe_question_runs_one_plan_and_one_grounded_answer(self):
        plan = {'scope': 'in_domain', 'direct_response': '', 'use_previous_context': False,
                'goal': 'Recommend good anime from the 90s', 'clarification': '', 'steps': [
                    {'tool': 'recommend_anime', 'arguments': {'year_from': 1990, 'year_to': 1999, 'sort_by': 'score'}}]}
        prompts = []
        def reply(prompt, *_):
            prompts.append(prompt)
            return 'Show 3 (1995), Show 4 (1999), and Show 2 (1990) have the highest scores.'
        models = SimpleNamespace(embed=lambda texts: np.ones((len(texts), 2), dtype=np.float32),
                                 cache_revision='test', plan_tools=lambda *_: json.dumps(plan), agent_reply=reply)
        settings = replace(load_settings(), documents=self.folder, index=self.folder / 'index')
        agent = EntityAgent(settings, models, self.catalog)
        self.assertIn('Show 3', agent.answer('What good anime came out at the 90s', []))
        self.assertEqual(agent.last_usage['llm_calls'], 2)
        self.assertEqual(agent.last_usage['executed_tools'], 1)
        self.assertEqual([r['id'] for r in agent.last_trace[0]['result']['items']], ['3', '4', '2'])
        self.assertNotIn('missing_recommendation_criteria', prompts[0])

    def test_schema_accepts_metadata_filters_and_rejects_invalid_types(self):
        plan = {'scope': 'in_domain', 'direct_response': '', 'use_previous_context': False,
                'goal': 'Find suitable titles.', 'clarification': '', 'steps': [{
                    'tool': 'recommend_anime', 'arguments': {'year_from': 1985, 'year_to': 2005,
                    'studio': 'Studio One', 'staff': 'Alice Example', 'staff_role': 'Director',
                    'source_material': 'MANGA', 'sort_by': 'score'}}]}
        self.assertEqual(validate_plan(json.dumps(plan), plan_schema(8)), plan)
        plan['steps'][0]['arguments']['year_from'] = 'the eighties'
        with self.assertRaisesRegex(ValueError, 'integer'):
            validate_plan(json.dumps(plan), plan_schema(8))

    def test_existing_exports_upgrade_metadata_without_touching_documents(self):
        old = dict(self.records[0])
        for key in ('metadata_schema', 'source_material', 'studios', 'staff'):
            old.pop(key)
        path = self.folder / 'anime-1.md'
        text = '# Show 1\nsource: MANGA\nstudios: [{"node":{"name":"Legacy Studio"}}]\n'
        path.write_text(text, encoding='utf-8')
        (self.folder / 'catalog.jsonl').write_text(json.dumps(old), encoding='utf-8')
        upgraded = AnimeTools.load(self.folder)
        self.assertEqual(upgraded.record('1')['source_material'], 'MANGA')
        self.assertEqual(upgraded.popular(5, {'studio': 'Legacy Studio'})[0]['id'], '1')
        self.assertEqual(path.read_text(encoding='utf-8'), text)
        # Once indexed, metadata is read from SQL rather than rescanning exports.
        with patch.object(Path, 'open', side_effect=AssertionError('must reuse metadata')):
            self.assertEqual(AnimeTools.load(self.folder).record('1')['source_material'], 'MANGA')


if __name__ == '__main__':
    unittest.main()
