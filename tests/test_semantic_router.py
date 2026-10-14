import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config import load_settings
from src.semantic_router import SemanticRouter


class SemanticRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = replace(load_settings(), index=Path(self.temp.name), router_top_k=2)
        self.prompts, self.embeddings = [], []
        self.output = {'intent': 'cast', 'subject': 'Higurashi', 'field': None, 'limit': 5,
                       'confidence': .9, 'entity_type': 'anime'}
        def embed(texts):
            self.embeddings.append(texts)
            return np.eye(2, dtype=np.float32) if len(texts) == 2 else np.array([[1., 0.]], dtype=np.float32)
        def route_plan(prompt, **kwargs):
            self.prompts.append(prompt)
            return json.dumps(self.output)
        self.models = SimpleNamespace(embed=embed, route_plan=route_plan, cache_revision='test')
        self.router = SemanticRouter(self.settings, self.models)
        self.router.examples = [
            {'intent': 'cast', 'description': 'Fictional protagonists of one anime', 'examples': ['Who carries the story?']},
            {'intent': 'fact', 'field': 'episodes', 'description': 'Episode count', 'examples': ['How many installments?']}]
        self.tools = SimpleNamespace(titles=lambda q, limit: [{'id':'934', 'title':'When They Cry',
            'aliases':['When They Cry', 'Higurashi']}] if 'Higurashi' in q else [], character=lambda q: None)

    def test_retrieves_examples_then_generates_plan_for_paraphrase(self):
        plan = self.router.plan('Which folks carry the narrative in Higurashi?', [], self.tools)
        self.assertEqual(plan.intent, 'cast')
        self.assertEqual(plan.subject, 'Higurashi')
        self.assertIn('retrieved_tools', self.prompts[0])
        self.assertIn('Fictional protagonists', self.prompts[0])
        self.assertIn('ENTITY', self.embeddings[-1][0])
        self.assertEqual(self.router.last_decision['retrieved'][0][0], 'cast')

    def test_same_question_reuses_plan_but_history_changes_replan(self):
        self.router.plan('Cast of Higurashi?', [], self.tools)
        count = len(self.embeddings)
        self.router.plan('Cast of Higurashi?', [], self.tools)
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(len(self.embeddings), count)
        self.assertTrue(self.router.last_decision['cached'])
        self.router.plan('Cast of Higurashi?', [{'role':'user', 'content':'What is Higurashi?'}], self.tools)
        self.assertEqual(len(self.prompts), 2)

    def test_history_is_supplied_to_planner_for_omitted_subject(self):
        plan = self.router.plan('Who carries the story?', [{'role':'user', 'content':'What is Higurashi?'}], self.tools)
        self.assertEqual(plan.subject, 'Higurashi')
        self.assertIn('What is Higurashi?', self.prompts[0])

    def test_rejects_invented_subject_tool_field_and_invalid_confidence(self):
        invalid = [{'subject':'Invented unrelated anime'}, {'intent':'execute_shell'},
                   {'intent':'fact', 'field':'password'}, {'confidence': True}, {'limit': 500}, {'entity_type':'system'},
                   {'subject':'the main character'}]
        for change in invalid:
            with self.subTest(change=change):
                self.router.cache.clear()
                original = dict(self.output)
                self.output.update(change)
                with self.assertRaises(RuntimeError):
                    self.router.plan('Cast of Higurashi?', [], self.tools)
                self.output = original

    def test_confidence_does_not_override_model_selected_route(self):
        self.output['confidence'] = .3
        self.assertEqual(self.router.plan('Cast of Higurashi?', [], self.tools).intent, 'cast')

    def test_malformed_json_never_executes_a_tool(self):
        self.models.route_plan = lambda prompt, **kwargs: 'not JSON'
        with self.assertRaises(RuntimeError):
            self.router.plan('Cast of Higurashi?', [], self.tools)

    def test_invalid_plan_is_repaired_by_model_not_replaced_with_clarification(self):
        responses = iter(['not JSON', json.dumps(dict(self.output, intent='detail'))])
        self.models.route_plan = lambda prompt, **kwargs: self.prompts.append(prompt) or next(responses)
        self.assertEqual(self.router.plan('Cast of Higurashi?', [], self.tools).intent, 'detail')
        self.assertIn('validation_error', self.prompts[-1])
        self.assertEqual(self.router.last_decision['validation_retries'], 1)

    def test_model_can_select_available_tool_absent_from_retrieved_examples(self):
        self.output.update(intent='fact', field='duration')
        plan = self.router.plan('Cast of Higurashi?', [], self.tools)
        self.assertEqual(plan.field, 'duration')

    def test_episode_follow_up_retains_title_from_previous_fact_question(self):
        self.output.update(intent='fact', subject='Higurashi', field='episodes')
        history = [{'role':'user', 'content':'How long is Higurashi?'},
                   {'role':'assistant', 'content':'An episode is 24 minutes.'}]
        plan = self.router.plan('How many episodes are there?', history, self.tools)
        self.assertEqual((plan.intent, plan.subject, plan.field), ('fact', 'Higurashi', 'episodes'))

    def test_recommendation_follow_up_uses_explicit_seed_and_constrained_schema(self):
        self.output.update(intent='recommend', subject='Suggest anime like that', seed_titles=['When They Cry'])
        schemas = []
        def model(prompt, **kwargs):
            schemas.append(kwargs['schema'])
            return json.dumps(self.output)
        self.models.route_plan = model
        history = [{'role':'user', 'content':'What is Higurashi about?'}]
        plan = self.router.plan('Suggest anime like that', history, self.tools)
        self.assertEqual(plan.seed_titles, ('When They Cry',))
        self.assertEqual(schemas[0]['properties']['seed_titles']['items']['enum'], ['When They Cry'])

    def test_all_bounded_history_reaches_planner_not_just_last_three_turns(self):
        history = [{'role':'user', 'content':'What is Higurashi?'}]
        history += [{'role':'user', 'content':'And what about that?'} for _ in range(7)]
        self.router.plan('Who carries the story?', history, self.tools)
        self.assertIn('What is Higurashi?', self.prompts[-1])

    def test_route_embeddings_reused_across_router_instances(self):
        self.router.plan('Cast of Higurashi?', [], self.tools)
        self.embeddings.clear()
        second = SemanticRouter(self.settings, self.models)
        second.examples = self.router.examples
        second.plan('Cast of Higurashi?', [], self.tools)
        self.assertEqual(len(self.embeddings), 1)

    def test_assistant_hallucination_cannot_authorize_a_different_character(self):
        self.output.update(intent='character', subject='Lelouch Lamperouge', entity_type='character')
        history = [{'role':'user', 'content':'Who is Natsuki Subaru?'},
                   {'role':'assistant', 'content':'Lelouch Lamperouge from Code Geass.'}]
        with self.assertRaises(ValueError):
            self.router.validate(json.dumps(self.output), 'I mean from Re:Zero', history, [],
                                 [{'intent':'character'}])

    def test_correction_keeps_user_name_and_adds_anime_scope(self):
        self.output.update(intent='character', subject='Natsuki Subaru', entity_type='character', anime_scope='Re:Zero')
        history = [{'role':'user', 'content':'Who is Natsuki Subaru?'},
                   {'role':'assistant', 'content':'A different Subaru works at a cafe.'}]
        plan, _ = self.router.validate(json.dumps(self.output), 'I mean from Re:Zero', history, [],
                                       [{'intent':'character'}])
        self.assertEqual(plan.subject, 'Natsuki Subaru')
        self.assertEqual(plan.anime_scope, 'Re:Zero')

    def test_known_full_name_can_supply_its_catalog_anime_association(self):
        self.output.update(intent='character', subject='Natsuki Subaru', entity_type='character', anime_scope='Re:Zero')
        candidates = [{'title':'Subaru Natsuki', 'aliases':['Subaru Natsuki'],
                       'entity_type':'character', 'anime':'Re:Zero -Starting Life in Another World-'}]
        plan, _ = self.router.validate(json.dumps(self.output), 'Who is Natsuki Subaru?', [], candidates,
                                       [{'intent':'character'}])
        self.assertEqual(plan.anime_scope, 'Re:Zero')

    def test_detailed_character_request_cannot_take_identity_from_assistant(self):
        self.output.update(intent='detail', subject='Lelouch Lamperouge', entity_type='character')
        history = [{'role':'user', 'content':'Who is Natsuki Subaru?'},
                   {'role':'assistant', 'content':'Lelouch Lamperouge from Code Geass.'}]
        with self.assertRaises(ValueError):
            self.router.validate(json.dumps(self.output), 'I mean from Re:Zero', history, [],
                                 [{'intent':'detail'}])


if __name__ == '__main__':
    unittest.main()
