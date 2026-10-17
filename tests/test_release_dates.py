"""Regression coverage for the failed ISO-date request and existing year behavior."""
import json
import unittest
from types import SimpleNamespace

import test_metadata_recommendations as fixtures
from src.catalog_search import BASE_FIELDS, predicates
from src.config import load_settings
from src.entity_agent import EntityAgent
from src.release_dates import complete_date, release_date_filters
from src.request_constraints import catalog_text_values, merge_planner_attributes
from src.request_interpreter import FIELD_MEANINGS
from src.semantic_attributes import interpret_attributes


class ReleaseDateParsingTests(unittest.TestCase):
    def filters(self, question):
        return [(f['operator'], f['value']) for f in release_date_filters(question)[0]]

    def test_original_full_date_range_is_inclusive_and_not_projected_to_years(self):
        self.assertEqual(self.filters('Recommend three anime with a first release date from 2018-01-01 through 2019-12-31.'),
                         [('gte', '2018-01-01'), ('lte', '2019-12-31')])
        self.assertEqual(self.filters('Anime between 2018-06-15 and 2018-08-20'),
                         [('gte', '2018-06-15'), ('lte', '2018-08-20')])

    def test_open_exclusive_and_negated_comparisons_preserve_boundaries(self):
        for phrase, expected in [('after', 'gt'), ('before', 'lt'), ('since', 'gte'),
                                 ('on or after', 'gte'), ('on or before', 'lte'),
                                 ('not before', 'gte'), ('not after', 'lte'), ('not on', 'ne'), ('on', 'eq')]:
            with self.subTest(phrase=phrase):
                self.assertEqual(self.filters(f'Anime released {phrase} 2018-06-15'), [(expected, '2018-06-15')])
        self.assertEqual(self.filters('Anime after 2018-06-15 and before 2018-08-20'),
                         [('gt', '2018-06-15'), ('lt', '2018-08-20')])
        self.assertEqual(self.filters('Anime after 2018-06-15 through 2018-08-20'),
                         [('gt', '2018-06-15'), ('lte', '2018-08-20')])

    def test_bad_or_impossible_dates_never_silently_relax_to_year_filters(self):
        for question in ['Anime from 2019-02-29 through 2019-12-31',
                         'Anime from 2019-12-31 to 2018-01-01',
                         'Anime not between 2018-06-15 and 2018-08-20',
                         'Anime after 2018-06-15 and before 2018-06-15']:
            with self.subTest(question=question), self.assertRaises(ValueError):
                release_date_filters(question)
        self.assertIsNone(complete_date(2018, None, None))
        self.assertIsNone(complete_date(2019, 2, 29))
        self.assertEqual(complete_date(2020, 2, 29), '2020-02-29')

    def test_years_decades_and_other_numeric_attributes_keep_the_existing_resolver(self):
        for question in ['Anime from 2018 through 2019', 'Anime from the nineties', 'Anime after 2015']:
            self.assertEqual(release_date_filters(question), ([], []))
        def model(prompt, schema):
            if 'operator' in schema['properties']:
                return {'operator': schema['properties']['operator']['enum'][0]}
            return {'min': 2018, 'max': 2019}
        result = interpret_attributes(model, 'Anime from 2018 through 2019', BASE_FIELDS, FIELD_MEANINGS,
                                      {'year': {'sources': ['2018 through 2019'], 'interpretations': ['range']}})
        self.assertEqual([(f['field'], f['operator'], f['value']) for f in result[0]],
                         [('year', 'gte', '2018'), ('year', 'lte', '2019')])

    def test_iso_dates_do_not_call_the_numeric_year_model(self):
        def unexpected(*_):
            self.fail('Full ISO bounds must not be reinterpreted as numeric years')
        result = interpret_attributes(unexpected, 'Anime from 2018-01-01 through 2019-12-31', BASE_FIELDS, FIELD_MEANINGS,
                                      {'year': {'sources': ['release'], 'interpretations': ['range']}})
        self.assertEqual([f['field'] for f in result[0]], ['start_date', 'start_date'])


class ReleaseDateRetrievalTests(unittest.TestCase):
    def setUp(self):
        fixtures.MetadataRecommendationTests.setUp(self)
        # Same-year fixtures prove that month/day boundaries are really enforced.
        dates = {'1': (2018, 6, 14), '2': (2018, 6, 15), '3': (2018, 8, 20),
                 '4': (2018, 8, 21), '5': (2018, None, None), '6': (2019, 2, 29)}
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.catalog.database)) as db:
            for identifier, parts in dates.items():
                metadata = json.loads(db.execute('SELECT data FROM anime_metadata WHERE anime_id=?', (identifier,)).fetchone()[0])
                metadata.setdefault('search_data', {}).update(dict(zip(
                    ('startDate_year', 'startDate_month', 'startDate_day'), parts)))
                db.execute('UPDATE anime_metadata SET data=? WHERE anime_id=?', (json.dumps(metadata), identifier))
            db.commit()

    def recommend(self, **constraints):
        return self.store.recommendations([], [], 20, constraints)['items']

    def test_exact_dates_include_endpoints_and_exclude_partial_invalid_and_adjacent_dates(self):
        filters, _ = release_date_filters('Anime from 2018-06-15 through 2018-08-20')
        actual = [{k: f[k] for k in ('field', 'operator', 'value')} for f in filters]
        items = self.recommend(filters=actual)
        self.assertEqual({r['id'] for r in items}, {'2', '3'})
        self.assertEqual({r['start_date'] for r in items}, {'2018-06-15', '2018-08-20'})
        self.assertEqual({r['matched_fields']['start_date'] for r in items}, {'2018-06-15', '2018-08-20'})
        actual[0]['operator'] = 'gt'
        self.assertEqual([r['id'] for r in self.recommend(filters=actual)], ['3'])

    def test_date_predicates_reject_numeric_years_and_bad_dates(self):
        for value in ('2018', '2018-02-30', '2018-06-15 OR 1=1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'ISO calendar date'):
                predicates([{'field': 'start_date', 'operator': 'gte', 'value': value}], self.catalog.searchable_fields())

    def test_malformed_planner_years_cannot_break_authoritative_date_or_numeric_filters(self):
        question = 'Recommend anime from 2018-06-15 through 2018-08-20'
        args = {'filters': [{'field': 'year', 'operator': 'gt', 'value': '2018-06-15'},
                            {'field': 'year', 'operator': 'lte', 'value': '2018-08-20'}]}
        dates, _ = release_date_filters(question)
        request = {'intent': 'recommendation', 'filters': [{k: f[k] for k in ('field', 'operator', 'value')} for f in dates],
                   'soft_query': '', 'mappings': [], 'limitations': [], 'entity_queries': [], 'use_previous_context': False}
        before = list(request['filters'])
        merge_planner_attributes(request, args, question, self.catalog.searchable_fields(), {})
        self.assertEqual(request['filters'], before)
        agent = EntityAgent(load_settings(), SimpleNamespace(), self.catalog)
        agent._constraint_text_values = catalog_text_values(self.catalog, self.catalog.searchable_fields())
        _, state, _, _ = agent.prepare(question, [], request)
        state['request_question'] = question
        result = agent.execute('recommend_anime', args, state)
        agent.verify_recommendation(result, state)
        self.assertEqual({item['id'] for item in result['items']}, {'2', '3'})
        numeric = {'filters': [{'field': 'year', 'operator': 'gte', 'value': '2018'}], 'mappings': []}
        merge_planner_attributes(numeric, args, question, BASE_FIELDS, {})
        self.assertEqual(numeric['filters'], [{'field': 'year', 'operator': 'gte', 'value': '2018'}])
        merge_planner_attributes(numeric, {'filters': [{'field': 'start_date', 'operator': 'gt', 'value': '2018'}]},
                                 'Anime since 2018', BASE_FIELDS, {})
        self.assertEqual(numeric['filters'], [{'field': 'year', 'operator': 'gte', 'value': '2018'}])
