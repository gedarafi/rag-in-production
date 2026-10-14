import json
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.anime_tools import AnimeTools, structured
from src.config import load_settings
from src.entity_agent import EntityAgent
from src.entity_store import EntityStore


def call(tool, **args):
    return {'action': 'call', 'tool': tool, 'arguments': args, 'text': ''}


def answer(text):
    return {'action': 'answer', 'tool': None, 'arguments': {}, 'text': text}


class EntityAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'anilist'
        self.folder.mkdir()
        rows = [
            {'id': '1', 'title_english': 'Re:Zero', 'synonyms': ['ReZero'], 'popularity': 500,
             'episodes': 25, 'duration': 24, 'genres': ['Fantasy', 'Sci-Fi'], 'tags': [{'name': 'Time Loop'}],
             'description': 'Subaru is transported to another world and returns after death.',
             'characters': [{'role': 'MAIN', 'node': {'id': 17, 'name': {'full': 'Subaru Natsuki'},
                              'description': 'An ordinary transported boy. He returns after death and protects his friends.'}}],
             'recommendations': [{'node': {'rating': 40, 'mediaRecommendation': {'id': '2'}}}]},
            {'id': '2', 'title_english': 'Other World', 'popularity': 400, 'genres': ['Fantasy'],
             'description': 'Another ordinary boy is transported to another world.',
             'characters': [{'role': 'MAIN', 'node': {'id': 18, 'name': {'full': 'Another Hero'},
                              'description': 'An ordinary transported boy who returns after death and protects his friends.'}}]},
            {'id': '3', 'title_english': 'Unrelated Show', 'popularity': 1000, 'genres': ['Comedy'],
             'description': 'A supporting character has similar traits.',
             'characters': [{'role': 'SUPPORTING', 'node': {'id': 19, 'name': {'full': 'Supporting Hero'},
                              'description': 'An ordinary transported boy who returns after death and protects his friends.'}}]},
        ]
        records = [structured(row, f"anime-{row['id']}.md") for row in rows]
        (self.folder / 'catalog.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        self.catalog = AnimeTools.load(self.folder)
        self.store = EntityStore(self.catalog)
        self.prompts, self.decisions = [], []
        def step(prompt, schema):
            self.prompts.append((prompt, schema))
            return json.dumps(self.decisions.pop(0))
        def embed(texts):
            return np.array([[1., 0.] if 'transported' in text.lower() else [0., 1.] for text in texts], dtype=np.float32)
        self.models = SimpleNamespace(embed=embed, agent_step=step, cache_revision='test')
        settings = replace(load_settings(), documents=self.folder, index=self.root / 'index', agent_max_steps=8)
        self.agent = EntityAgent(settings, self.models, self.catalog)
        # Regression coverage for the retired iterative protocol. Production
        # two-call behavior is tested separately in test_tool_plan.py.
        self.agent.answer = self.agent._answer_iterative

    def test_separate_dictionaries_handle_typos_and_name_order(self):
        self.assertEqual(self.store.lookup('anime', 'ReZro')['matches'][0]['id'], '1')
        self.assertEqual(self.store.lookup('character', 'Natsuki Subaru')['matches'][0]['id'], '17')
        self.assertEqual(self.store.lookup('genre', 'SciFi')['matches'][0]['id'], 'sci fi')
        self.assertEqual(self.store.lookup('tag', 'Time Lop')['matches'][0]['id'], 'time loop')

    def test_full_bundle_answers_multiple_fields_without_vector_search(self):
        self.decisions = [call('get_anime', id='1'), answer('Re:Zero has 25 episodes, each 24 minutes. [anime-1.md]')]
        result = self.agent.answer('How long is Re:Zero?', [])
        self.assertIn('25 episodes', result)
        bundle_prompt = self.prompts[-1][0]
        self.assertIn('"episodes": 25.0', bundle_prompt)
        self.assertIn('"duration": 24.0', bundle_prompt)
        self.assertIn('"main_characters"', bundle_prompt)
        self.assertEqual([r['tool'] for r in self.agent.last_trace], ['get_anime'])

    def test_conversation_scoped_ids_supply_follow_up_bundle(self):
        first = 'Re:Zero episodes last 24 minutes. [anime-1.md]'
        self.decisions = [call('get_anime', id='1'), answer(first)]
        self.agent.answer('How long is Re:Zero?', [])
        self.decisions = [answer('There are 25 episodes. [anime-1.md]')]
        history = [{'role': 'user', 'content': 'How long is Re:Zero?'}, {'role': 'assistant', 'content': first}]
        self.assertIn('25', self.agent.answer('How many episodes are there?', history))
        self.assertIn('"active_anime_ids": ["1"]', self.prompts[-1][0])
        self.assertIn('"episodes": 25.0', self.prompts[-1][0])
        self.decisions = [{'action': 'clarify', 'tool': None, 'arguments': {}, 'text': 'Which series?'}]
        self.assertEqual(self.agent.answer('How many episodes are there?', []), 'Which series?')
        self.assertIn('"active_anime_ids": []', self.prompts[-1][0])

    def test_layered_anime_cast_character_recommendation_calls(self):
        self.decisions = [call('get_anime', id='1'), call('get_character', id='17'),
                          call('recommend_by_character', character_ids=['17'], query='ordinary transported boy returns after death'),
                          answer('Try Other World: its main character has a similar situation. [anime-2.md]')]
        result = self.agent.answer('Anime with a main character like Re:Zero?', [])
        self.assertIn('Other World', result)
        self.assertEqual([r['tool'] for r in self.agent.last_trace], ['get_anime', 'get_character', 'recommend_by_character'])
        self.assertIn('"matching_character"', self.prompts[-1][0])
        self.assertNotIn('"title": "Unrelated Show"', self.prompts[-1][0])

    def test_lookup_can_correct_typo_then_load_selected_entity(self):
        self.decisions = [call('resolve_entity', kind='character', query='Natski Subaru'),
                          call('get_character', id='17'), answer('Subaru Natsuki is from Re:Zero. [anime-1.md]')]
        self.assertIn('Subaru Natsuki', self.agent.answer('Who is Natski Subaru?', []))
        self.assertIn('fuzzy', self.prompts[1][0])

    def test_invalid_ids_are_feedback_for_model_not_global_fallback(self):
        self.decisions = [call('get_anime', id='999'), call('get_anime', id='1'), answer('Re:Zero. [anime-1.md]')]
        self.agent.answer('What is Re:Zero?', [])
        self.assertIn('validation_feedback', self.prompts[1][0])
        self.assertIn('Use anime IDs', self.prompts[1][0])

    def test_reference_topic_survives_recommendations(self):
        first = 'Try Other World if you liked Re:Zero. [anime-2.md]'
        self.decisions = [call('get_anime', id='1'), call('recommend_anime', seed_anime_ids=['1']), answer(first)]
        self.agent.answer('Recommend anime like Re:Zero', [])
        history = [{'role':'user', 'content':'Recommend anime like Re:Zero'}, {'role':'assistant', 'content':first}]
        self.decisions = [answer('The reference anime has 25 episodes. [anime-1.md]')]
        self.agent.answer('How many episodes does that reference have?', history)
        self.assertIn('"active_anime_ids": ["1"]', self.prompts[-1][0])

    def test_large_character_profiles_are_explicitly_paged(self):
        with closing(__import__('sqlite3').connect(self.catalog.database)) as db:
            row = json.loads(db.execute("SELECT data FROM characters WHERE id='17'").fetchone()[0])
            row['description'] = 'Long profile. ' * 1000
            db.execute("UPDATE characters SET data=? WHERE id='17'", (json.dumps(row),))
            db.commit()
        first = self.store.character('17', limit=1000)
        self.assertEqual(first['next_offset'], 1000)
        self.assertEqual(len(first['description']), 1000)
        self.assertGreater(first['description_characters'], 1000)

    def test_factual_answer_cannot_skip_evidence(self):
        self.decisions = [answer('It has 99 episodes.'), call('resolve_entity', kind='anime', query='ReZro'),
                          call('get_anime', id='1'),
                          answer('It has 25 episodes. [anime-1.md]')]
        self.assertIn('25 episodes', self.agent.answer('How many episodes in ReZro?', []))
        self.assertNotIn('answer', [b['properties']['action']['const'] for b in self.prompts[0][1]['anyOf']])
        self.assertIn('requires loaded source evidence', self.prompts[1][0])

    def test_selected_citation_is_rendered_as_metadata(self):
        response = answer('Re:Zero has 25 episodes.')
        response['citations'] = ['anime-1.md']
        self.decisions = [call('get_anime', id='1'), response]
        self.assertEqual(self.agent.answer('How long is Re:Zero?', []),
                         'Re:Zero has 25 episodes. [anime-1.md]')
        branch = next(b for b in self.prompts[-1][1]['anyOf'] if b['properties']['action']['const'] == 'answer')
        self.assertIn('anime-1.md', branch['properties']['citations']['items']['enum'])

    def test_facts_cannot_be_smuggled_in_as_clarification(self):
        self.decisions = [{'action': 'clarify', 'text': 'It has 99 episodes.'},
                          call('get_anime', id='1'), answer('It has 25 episodes. [anime-1.md]')]
        self.assertIn('25 episodes', self.agent.answer('How long is Re:Zero?', []))
        self.assertIn('Clarification must be a concise question', self.prompts[1][0])

    def test_unique_explicit_title_can_answer_without_planning_calls(self):
        self.decisions = [answer('Re:Zero has 25 episodes. [anime-1.md]')]
        self.assertIn('25 episodes', self.agent.answer('How many episodes in Re:Zero?', []))
        self.assertEqual(self.agent.last_trace, [])

    def test_follow_up_recovers_user_subject_after_process_cache_loss(self):
        history = [{'role': 'user', 'content': 'What is Re:Zero?'},
                   {'role': 'assistant', 'content': 'A previous potentially unreliable synopsis.'}]
        self.decisions = [answer('There are 25 episodes. [anime-1.md]')]
        self.assertIn('25 episodes', self.agent.answer('How many episodes are there?', history))
        self.assertIn('"episodes": 25.0', self.prompts[-1][0])

    def test_budget_exhaustion_gets_natural_llm_answer(self):
        self.agent.settings = replace(self.agent.settings, agent_max_steps=1)
        self.decisions = [call('get_anime', id='1')]
        final_prompts = []
        def finish(prompt, temperature, max_tokens):
            final_prompts.append(prompt)
            return 'Re:Zero has 25 episodes. [anime-1.md]'
        self.models.answer = finish
        self.assertIn('25 episodes', self.agent.answer('How long is Re:Zero?', []))
        self.assertIn('"episodes": 25.0', final_prompts[0])
        self.assertNotIn('Return JSON', final_prompts[0])

    def test_repeated_invalid_calls_stop_and_llm_clarifies(self):
        self.decisions = [call('get_anime', id='999'), call('get_anime', id='999')]
        self.models.answer = lambda *args: 'Which series did you have in mind?'
        self.assertEqual(self.agent.answer('How long is it?', []), 'Which series did you have in mind?')
        self.assertEqual(len(self.prompts), 2)

    def test_appearance_page_still_includes_character_biography(self):
        self.decisions = [call('resolve_entity', kind='character', query='Natski Subaru'),
                          call('get_character', id='17', section='appearances'),
                          answer('Subaru Natsuki is an ordinary transported boy. [anime-1.md]')]
        self.agent.answer('Who is Natski Subaru?', [])
        self.assertIn('ordinary transported boy', self.prompts[-1][0])

    def test_new_character_overrides_unrelated_history_bundle(self):
        history = [{'role': 'user', 'content': 'What is Unrelated Show?'},
                   {'role': 'assistant', 'content': 'An old topic.'}]
        self.decisions = [answer('Subaru Natsuki is an ordinary transported boy. [anime-1.md]')]
        self.agent.answer('Who is Natski Subaru?', history)
        self.assertIn('ordinary transported boy', self.prompts[0][0])
        self.assertNotIn('A supporting character has similar traits', self.prompts[0][0])

    def test_character_dictionary_can_be_constrained_to_named_anime(self):
        matches = self.store.lookup('character', 'Hero', anime_ids=['2'])['matches']
        self.assertEqual([m['id'] for m in matches], ['18'])

    def test_grouped_citations_are_valid_provenance(self):
        self.assertEqual(self.agent.format_citations('Sources [anime-1.md, anime-2.md]',
                                                   {'anime-1.md', 'anime-2.md'}),
                         'Sources [anime-1.md] [anime-2.md]')

    def test_character_introduction_cannot_be_only_a_name(self):
        self.decisions = [answer('Subaru Natsuki [anime-1.md]'),
                          answer('Subaru Natsuki is an ordinary transported boy who returns after death. [anime-1.md]')]
        self.assertIn('returns after death', self.agent.answer('Who is Natski Subaru?', []))
        self.assertIn('only repeats the character name', self.prompts[1][0])


if __name__ == '__main__':
    unittest.main()
