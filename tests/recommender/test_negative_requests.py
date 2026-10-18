import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch

from tests.recommender import test_tool_plan as fixtures
from tests.recommender.test_tool_plan import step


class NegativeRequestTests(unittest.TestCase):
    def setUp(self):
        fixtures.ToolPlanTests.setUp(self)
        self.attribute_retrieval = patch('src.recommender.request_resolution.retrieve_attributes', return_value={})
        self.attribute_retrieval.start()
        self.addCleanup(self.attribute_retrieval.stop)
        self.request = {'intent': 'recommendation', 'use_previous_context': False,
                        'entity_queries': [], 'filters': [], 'soft_query': '', 'limitations': []}
        def interpret(prompt, _schema):
            if 'intent' in _schema['properties']:
                return json.dumps({k: self.request[k] for k in ('intent', 'use_previous_context', 'entity_queries')} | {
                    'required': [f['value'] for f in self.request['filters'] if f['operator'] in ('eq', 'contains')],
                    'excluded': [f['value'] for f in self.request['filters'] if f['operator'] in ('ne', 'not_contains')],
                    'preferred': []})
            if 'role' in _schema['properties'] and 'choice' not in _schema['properties']:
                phrase = prompt.rsplit('CONCEPT: ', 1)[-1]
                target = next((f for f in self.request['filters'] if f['value'].lower() == phrase.lower()), None)
                return json.dumps({'role': ('excluded' if target['operator'] in ('ne', 'not_contains') else 'required')
                                   if target else 'irrelevant'})
            return json.dumps({'choice': 0, 'role': 'irrelevant', 'operator': 'eq', 'value': '', 'mapping': 'exact'})
        self.models.interpret_request = interpret

    def ask(self, question):
        text = self.agent.answer(question, [])
        self.assertEqual(self.calls, ['plan', 'final'])
        self.assertGreaterEqual(self.agent.last_usage['llm_calls'], 3)
        return text

    def test_false_title_and_wrong_planner_constraints_cannot_override_contract(self):
        self.reply = 'These titles match the catalog filters; minor subplots may still occur.'
        with closing(sqlite3.connect(self.catalog.database)) as db:
            db.execute("UPDATE anime SET title='Romance', data=json_set(data,'$.title','Romance','$.aliases',json('[\"Romance\"]'),'$.genres',json('[\"Fantasy\",\"Romance\"]')) WHERE id='2'")
            db.execute("INSERT INTO aliases VALUES ('Romance','2')")
            db.execute("INSERT INTO features VALUES ('2','romance','Romance')")
            db.execute("INSERT INTO entity_dictionary VALUES ('genre','romance','Romance','[\"Romance\"]')")
            db.commit()
        question = "I'd like a Fantasy Anime that doesn't have romance"
        self.request['filters'] = [
            {'field': 'genres', 'operator': 'eq', 'value': 'Fantasy', 'source_text': 'Fantasy', 'mapping': 'exact'},
            {'field': 'tags', 'operator': 'not_contains', 'value': 'Romance', 'source_text': 'romance', 'mapping': 'exact'}]
        self.plan['steps'] = [step('recommend_anime', filters=[
            {'field': 'description', 'operator': 'contains', 'value': 'emotional'}])]
        self.ask(question)
        self.assertEqual(self.agent.last_usage['selected_context']['anime'], [])
        result = self.agent.last_trace[0]['result']
        self.assertNotIn('2', [r['id'] for r in result['items']])
        self.assertEqual(result['applied_filters']['filters'], [
            {'field': 'genres', 'operator': 'eq', 'value': 'Fantasy'},
            {'field': 'genres', 'operator': 'ne', 'value': 'Romance'}])
        self.assertNotIn('"tool": "entity_context"', self.final_prompts[-1])

    def test_recommendation_generation_uses_interpreted_intent(self):
        self.plan['steps'] = [step('recommend_anime')]
        self.models.agent_fact_reply = lambda *_args, **_kwargs: self.fail('Used factual generation')
        self.ask("I'd like a fantasy anime")
        self.assertIn('recommendation_evidence_missing', self.final_prompts[-1])

    def test_named_title_matching_is_independent_of_feature_words(self):
        _, _, _, current = self.agent.prepare('Recommend Fantasy anime like "Re:Zero"', [],
                                              {**self.request, 'entity_queries': ['Re:Zero']})
        self.assertEqual([r['id'] for r in current], ['1'])

    def test_missing_recommendation_step_is_supplied_from_validated_request(self):
        self.plan['steps'] = []
        self.ask('Something adventurous to watch please')
        self.assertEqual(self.agent.last_trace[-1]['tool'], 'recommend_anime')

    def test_exclusion_overclaim_gets_a_counted_grounding_correction(self):
        self.request['filters'] = [{'field': 'genres', 'operator': 'ne', 'value': 'Fantasy',
                                    'source_text': 'Fantasy', 'mapping': 'exact'}]
        self.plan['steps'] = [step('recommend_anime')]
        replies = iter(['These shows have no fantasy.', 'These titles match the catalog filters; minor subplots may still occur.'])
        self.models.agent_reply = lambda *_args: next(replies)
        self.assertIn('catalog filters', self.agent.answer('Recommend anime without Fantasy', []))
        self.assertTrue(self.agent.last_usage['exclusion_grounding_correction'])
        self.assertGreaterEqual(self.agent.last_usage['llm_calls'], 5)

    def test_verbose_candidate_metadata_cannot_evict_recommendation_evidence(self):
        original = self.agent.store.recommendations
        def verbose(*args, **kwargs):
            result = original(*args, **kwargs)
            for row in result['items']:
                row['aliases'] = ['A very long unused alias ' * 500] * 20
            return result
        self.agent.store.recommendations = verbose
        self.plan['steps'] = [step('recommend_anime')]
        self.ask('Recommend Fantasy anime')
        self.assertIn('"recommendation_evidence_missing": false', self.final_prompts[-1])
        self.assertIn('Re:Zero', self.final_prompts[-1])
