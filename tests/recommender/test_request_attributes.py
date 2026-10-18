import json
import unittest

from src.recommender.catalog_search import BASE_FIELDS
from src.recommender.request_constraints import merge_planner_attributes
from src.recommender.request_resolution import attribute_filters
from tests.recommender import test_metadata_recommendations as fixtures
from src.recommender.entity_agent import EntityAgent
from src.recommender.config import load_settings
from src.recommender.request_constraints import catalog_text_values
from types import SimpleNamespace


class RequestAttributeTests(unittest.TestCase):
    def test_attributes_are_extracted_independently_of_preference_hierarchy(self):
        def model(prompt, _schema):
            if set(_schema['properties']) == {'operator'}:
                return {'operator': _schema['properties']['operator']['enum'][0]}
            if 'applies' in _schema['properties'] and 'source_text' not in _schema['properties'] and 'value' not in _schema['properties']:
                return {'applies': 'when the anime was released' in prompt or 'how many episodes' in prompt}
            if 'role' in _schema['properties']:
                return {'role': 'exact'}
            if 'min' in _schema['properties']:
                episode = 'requested episode count' in prompt
                return {'min': None if episode else 2015, 'max': 12 if episode else None,
                        'min_inclusive': True, 'max_inclusive': True}
            payload = json.loads(prompt[prompt.index('{"field":'):])
            if payload['field'] == 'episodes':
                return {'applies': True, 'constraints': [{'operator': 'lte', 'value': '12'}]}
            if payload['field'] == 'year':
                return {'applies': True, 'constraints': [{'operator': 'gte', 'value': '2015'}]}
            if payload['field'] == 'format':
                return {'applies': True, 'operator': 'eq', 'value': 'TV'}
            return {'applies': False, 'constraints': []}
        result = attribute_filters(model, 'TV anime at most 12 episodes since 2015', BASE_FIELDS,
                                   {'format': ['TV', 'MOVIE']}, retrieved={
                                       'episodes': {'sources': ['at most 12 episodes'], 'interpretations': ['range']},
                                       'year': {'sources': ['since 2015'], 'interpretations': ['range']}})
        self.assertEqual({(f['field'], f['operator'], f['value']) for f in result},
                         {('episodes', 'lte', '12'), ('year', 'gte', '2015'), ('format', 'eq', 'TV')})

    def test_existing_numeric_constraints_survive_preference_compilation(self):
        request = {'filters': [{'field': 'genres', 'operator': 'ne', 'value': 'Romance'}], 'mappings': []}
        args = {'year_from': 2015, 'format': 'TV', 'filters': [
            {'field': 'episodes', 'operator': 'lte', 'value': '12'},
            {'field': 'description', 'operator': 'contains', 'value': 'emotional'}]}
        merge_planner_attributes(request, args, 'TV anime since 2015 with at most 12 episodes, emotional',
                                 BASE_FIELDS, {'format': ['TV', 'MOVIE']})
        self.assertEqual({f['field'] for f in request['filters']}, {'genres', 'year', 'episodes', 'format'})

    def test_planner_cannot_narrow_an_authoritative_inclusive_bound(self):
        request = {'filters': [{'field': 'episodes', 'operator': 'lte', 'value': '12'}], 'mappings': []}
        merge_planner_attributes(request, {'filters': [{'field': 'episodes', 'operator': 'lt', 'value': '12'}]},
                                 'at most 12 episodes', BASE_FIELDS, {})
        self.assertEqual(request['filters'], [{'field': 'episodes', 'operator': 'lte', 'value': '12'}])

    def test_unrequested_numbers_and_keyword_moods_are_not_adopted(self):
        request = {'filters': [], 'mappings': []}
        merge_planner_attributes(request, {'filters': [
            {'field': 'episodes', 'operator': 'lte', 'value': '12'},
            {'field': 'description', 'operator': 'contains', 'value': 'funny'}]},
                                 'Recommend something funny', BASE_FIELDS, {})
        self.assertEqual(request['filters'], [])

    def test_staff_role_still_matches_the_same_named_person(self):
        fixtures.MetadataRecommendationTests.setUp(self)
        request = {'intent': 'recommendation', 'entity_queries': [], 'filters': [], 'soft_query': '',
                   'limitations': [], 'mappings': [], 'use_previous_context': False}
        agent = EntityAgent(load_settings(), SimpleNamespace(), self.catalog)
        agent._constraint_text_values = catalog_text_values(self.catalog, self.catalog.searchable_fields())
        _, state, _, _ = agent.prepare('Anime directed by Alice Example', [], request)
        state['request_question'] = 'Anime directed by Alice Example'
        result = agent.execute('recommend_anime', {'staff': 'Alice Example', 'staff_role': 'Director'}, state)
        agent.verify_recommendation(result, state)
        self.assertNotIn('3', [item['id'] for item in result['items']])
        self.assertTrue(result['items'])
        self.assertEqual(result['applied_filters']['staff_role'], 'Director')
