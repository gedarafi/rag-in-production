import json
import sqlite3
import unittest
from contextlib import closing

from tests.recommender import test_entity_agent as fixtures
from src.recommender.anime_tools import normalize
from src.recommender.entity_agent import EntityAgent
from src.recommender.entity_resolution import resolve_mentions


class EntityResolutionTests(unittest.TestCase):
    def setUp(self):
        fixtures.EntityAgentTests.setUp(self)
        with closing(sqlite3.connect(self.catalog.database)) as db:
            for identifier, title in [('4', 'Air'), ('5', 'Original'), ('6', 'START'), ('7', 'Fantasy')]:
                record = dict(self.catalog.record('1'), id=identifier, title=title, aliases=[title], source=f'anime-{identifier}.md')
                db.execute('INSERT INTO anime(id,source,title,popularity,score,year,format,data) VALUES(?,?,?,?,?,?,?,?)',
                           (identifier, record['source'], title, 10, 50, 2000, 'TV', json.dumps(record)))
                db.execute('INSERT INTO aliases VALUES(?,?)', (normalize(title), identifier))
            db.execute("UPDATE meta SET value=value || '-typed-mentions' WHERE key='snapshot'")
            db.commit()
        self.store.ensure_dictionary()
        self.agent.last_usage = {'llm_calls': 0}
        self.decisions = []
        self.clarification = ''
        def decide(prompt, schema):
            slots = schema['properties']['decisions']['properties']
            decisions = {}
            for item in self.decisions:
                slot = 'p' + str(item['phrase_index'])
                if item['id']:
                    suffix = ' | ' + item['kind'] + ':' + item['id']
                    options = slots[slot]['properties']['selection']['enum']
                    selection = next((s for s in options if s.endswith(suffix)), 'forged' + suffix)
                else:
                    selection = 'ambiguous entity' if item['role'] == 'ambiguous' else 'unmapped description' if item['kind'] == 'description' else 'ordinary wording'
                decisions[slot] = {'meaning': 'Contextual classification.', 'selection': selection, 'role': item['role']}
            return json.dumps({'decisions': decisions, 'clarification': self.clarification})
        self.models.resolve_mentions = decide

    def resolve(self, question, queries, required=None, excluded=None):
        decomposition = dict(intent='factual', use_previous_context=False, entity_queries=queries,
                             required=required or [], excluded=excluded or [], preferred=[])
        return resolve_mentions(self.agent, question, [], decomposition)

    def decision(self, index, kind, identifier='', role='subject'):
        return dict(phrase_index=index, kind=kind, id=identifier, role=role)

    def prepare(self, question, resolution):
        request = dict(intent='factual', use_previous_context=False, entity_queries=[], filters=[], entity_resolution=resolution)
        return self.agent.prepare(question, [], request)

    def test_ordinary_words_never_expand_confirmed_subjects_or_promote_cast(self):
        question = 'When did the original Re:Zero first start to air?'
        self.assertGreater(len(self.catalog.titles(question, 10)), 1)  # old path has collisions
        self.decisions = [self.decision(0, 'anime', '1')]
        resolved = self.resolve(question, ['Re:Zero'])
        _, state, evidence, _ = self.prepare(question, resolved)
        self.assertEqual(state['active_anime_ids'], ['1'])
        self.assertEqual(state['active_character_ids'], [])
        self.assertEqual(state['confirmed_subjects'], [('anime', '1')])
        self.assertEqual([e['result']['id'] for e in evidence], ['1'])

    def test_common_words_are_still_genuine_titles_when_llm_selects_them(self):
        for title, identifier in [('Air', '4'), ('Original', '5'), ('START', '6')]:
            with self.subTest(title=title):
                self.decisions = [self.decision(0, 'anime', identifier)]
                resolved = self.resolve(f'What is {title} about?', [title])
                _, state, _, _ = self.prepare(f'What is {title} about?', resolved)
                self.assertEqual(state['active_anime_ids'], [identifier])

    def test_same_phrase_can_be_title_or_genre_without_blacklist(self):
        self.decisions = [self.decision(0, 'genre', 'fantasy', 'required')]
        genre = self.resolve('Recommend fantasy anime', [], required=['fantasy'])
        self.assertEqual(genre['subjects'], [])
        self.assertEqual(genre['mentions'][0]['kind'], 'genre')
        self.decisions = [self.decision(0, 'anime', '7')]
        title = self.resolve('Who animated Fantasy?', ['Fantasy'])
        self.assertEqual(title['subjects'][0]['id'], '7')

    def test_classification_precedes_lookup_and_restricts_candidate_types(self):
        from src.recommender.mention_interpreter import validate
        typed = dict(intent='recommendation', use_previous_context=False, mentions=[{
            'meaning': 'Desired genre', 'text': 'fantasy', 'query': 'Fantasy',
            'kind': 'genre', 'role': 'required'}])
        decomposition, mentions = validate(typed, 'Recommend fantasy anime', [])
        self.decisions = [self.decision(0, 'genre', 'fantasy', 'required')]
        result = resolve_mentions(self.agent, 'Recommend fantasy anime', [], decomposition, typed_mentions=mentions)
        self.assertEqual(result['subjects'], [])
        self.assertEqual(result['mentions'][0]['kind'], 'genre')
        # Even though a real anime called Fantasy exists, its ID is not eligible.
        self.decisions = [self.decision(0, 'anime', '7', 'required')]
        with self.assertRaisesRegex(ValueError, 'outside its candidates'):
            resolve_mentions(self.agent, 'Recommend fantasy anime', [], decomposition, typed_mentions=mentions)

    def test_typed_production_interpreter_keeps_ordinary_qualifiers_out_of_subjects(self):
        from src.recommender.request_resolution import interpret
        self.models.classify_mentions = lambda *_: json.dumps(dict(
            intent='factual', use_previous_context=False, mentions=[{
                'meaning': 'The work whose original adaptation is asked about', 'text': 'Re:Zero',
                'query': 'Re:Zero', 'kind': 'anime', 'role': 'subject'}]))
        self.decisions = [self.decision(0, 'anime', '1')]
        request = interpret(self.agent, 'When did the original Re:Zero first air?', [])
        _, state, _, _ = self.agent.prepare('When did the original Re:Zero first air?', [], request)
        self.assertEqual(state['confirmed_subjects'], [('anime', '1')])

    def test_classifier_cannot_invent_a_mention_from_outside_user_context(self):
        from src.recommender.mention_interpreter import validate
        typed = dict(intent='factual', use_previous_context=False, mentions=[{
            'meaning': 'Invented', 'text': 'Other World', 'query': 'Other World', 'kind': 'anime', 'role': 'subject'}])
        with self.assertRaisesRegex(ValueError, 'supported by the user request'):
            validate(typed, 'Tell me about Re:Zero', [])

    def test_unsourced_example_echo_cannot_become_a_subject(self):
        from src.recommender.mention_interpreter import validate
        real = {'meaning': 'Requested title', 'text': 'Re:Zero', 'query': 'Re:Zero', 'kind': 'anime', 'role': 'subject'}
        echo = {'meaning': 'An example, not this request', 'text': 'Other World', 'query': 'Other World', 'kind': 'anime', 'role': 'subject'}
        decomposition, mentions = validate(dict(intent='factual', use_previous_context=False, mentions=[real, echo]),
                                            'When did Re:Zero first air?', [])
        self.assertEqual(mentions, [real])
        self.assertEqual(decomposition['entity_queries'], ['Re:Zero'])

    def test_repeated_model_mentions_do_not_expand_scope_or_exceed_clause_limits(self):
        from src.recommender.mention_interpreter import validate
        real = {'meaning': 'Requested title', 'text': 'Original', 'query': 'Original', 'kind': 'anime', 'role': 'subject'}
        decomposition, mentions = validate(dict(intent='factual', use_previous_context=False, mentions=[real] * 16),
                                            'What is Original about?', [])
        self.assertEqual(mentions, [real])
        self.assertEqual(decomposition['entity_queries'], ['Original'])

    def test_descriptions_remain_unmapped_and_tags_are_not_subjects(self):
        self.decisions = [self.decision(0, 'tag', 'time loop', 'excluded'),
                          self.decision(1, 'description', '', 'preferred')]
        result = resolve_mentions(self.agent, 'No Time Loop, preferably comforting', [], dict(
            intent='recommendation', use_previous_context=False, entity_queries=[], required=[],
            excluded=['Time Loop'], preferred=['comforting']))
        self.assertEqual(result['subjects'], [])
        self.assertEqual(result['mentions'][1]['id'], '')

    def test_invented_ids_and_cross_type_ids_are_rejected(self):
        for kind, identifier in [('anime', 'invented'), ('character', '1')]:
            self.decisions = [self.decision(0, kind, identifier)]
            with self.assertRaisesRegex(ValueError, 'outside its candidates'):
                self.resolve('What is Re:Zero about?', ['Re:Zero'])

    def test_ambiguity_preserves_candidates_without_arbitrary_subject(self):
        self.decisions = [self.decision(0, 'anime', '', 'ambiguous')]
        self.clarification = 'Which adaptation do you mean?'
        result = self.resolve('Tell me about Re:Zero', ['Re:Zero'])
        self.assertEqual(result['subjects'], [])
        self.assertEqual(result['clarification'], self.clarification)

    def test_ambiguous_identity_asks_model_written_question_without_planning(self):
        self.decisions = [self.decision(0, 'anime', '', 'ambiguous')]
        self.clarification = 'Which adaptation do you mean?'
        self.models.interpret_request = lambda *_: json.dumps(dict(
            intent='factual', use_previous_context=False, entity_queries=['Re:Zero'],
            required=[], excluded=[], preferred=[]))
        self.models.plan_tools = lambda *_: self.fail('Ambiguity must not proceed to tool planning')
        result = EntityAgent.answer(self.agent, 'Tell me about Re:Zero', [])
        self.assertEqual(result, self.clarification)
        self.assertEqual(self.agent.last_usage['response_path'], 'entity_clarification')

    def test_unrelated_clarification_commentary_does_not_override_resolved_subject(self):
        self.decisions = [self.decision(0, 'anime', '1')]
        self.clarification = 'No clarification needed.'
        result = self.resolve('What is Re:Zero about?', ['Re:Zero'])
        self.assertEqual(result['clarification'], '')
        self.assertEqual(result['subjects'][0]['id'], '1')

    def test_factual_interpreter_uses_the_same_confirmed_resolution_as_recommendations(self):
        from src.recommender.request_resolution import interpret
        self.decisions = [self.decision(0, 'anime', '1')]
        self.models.interpret_request = lambda *_: json.dumps(dict(
            intent='factual', use_previous_context=False, entity_queries=['Re:Zero'],
            required=[], excluded=[], preferred=[]))
        request = interpret(self.agent, 'When did the original Re:Zero first air?', [])
        self.assertEqual(request['entity_queries'], ['Re:Zero'])
        _, state, _, _ = self.agent.prepare('When did the original Re:Zero first air?', [], request)
        self.assertEqual(state['confirmed_subjects'], [('anime', '1')])

    def test_referential_followup_rehydrates_confirmed_conversation_scope(self):
        self.decisions = [self.decision(0, 'anime', '1')]
        resolved = self.resolve('Tell me about Re:Zero', ['Re:Zero'])
        snapshot, state, _, _ = self.prepare('Tell me about Re:Zero', resolved)
        self.agent.remember(snapshot, [], 'Tell me about Re:Zero', 'An answer.', state, [])
        history = [{'role': 'user', 'content': 'Tell me about Re:Zero'}, {'role': 'assistant', 'content': 'An answer.'}]
        request = dict(intent='factual', use_previous_context=True, entity_queries=[], filters=[],
                       entity_resolution={'subjects': [], 'mentions': [], 'clarification': ''})
        _, followup, evidence, _ = self.agent.prepare('How many episodes does it have?', history, request)
        self.assertEqual(followup['active_anime_ids'], ['1'])
        self.assertEqual([e['result']['id'] for e in evidence], ['1'])

    def test_multiple_confirmed_entities_keep_their_types(self):
        self.decisions = [self.decision(0, 'anime', '1'), self.decision(1, 'character', '17')]
        resolved = self.resolve('Compare Re:Zero and Subaru Natsuki', ['Re:Zero', 'Subaru Natsuki'])
        _, state, _, _ = self.prepare('Compare Re:Zero and Subaru Natsuki', resolved)
        self.assertEqual(state['confirmed_subjects'], [('anime', '1'), ('character', '17')])
        self.assertEqual(state['active_anime_ids'], ['1'])
        self.assertEqual(state['active_character_ids'], ['17'])

    def test_tools_cannot_promote_related_entities_to_required_grounding_subjects(self):
        self.decisions = [self.decision(0, 'anime', '1')]
        resolved = self.resolve('How long is Re:Zero?', ['Re:Zero'])
        snapshot, state, evidence, current = self.prepare('How long is Re:Zero?', resolved)
        # A later tool fetches a cast profile. It is evidence, not a new user subject.
        character = self.agent.execute('get_character', {'id': '17'}, state)
        evidence.append({'tool': 'get_character', 'result': character})
        state['response_goal'] = 'Answer the question'
        self.agent.last_usage = {'llm_calls': 0}
        captured = []
        def reply(prompt, temperature, max_tokens, subjects, measurements, require_quantities=False):
            captured.extend(subjects)
            return '24 minutes per episode.'
        self.models.agent_fact_reply = reply
        self.models.agent_reply = lambda *_: 'unused'
        self.agent.finish('How long is Re:Zero?', [], state, current, evidence, [])
        self.assertEqual(captured, ['anime:1'])


if __name__ == '__main__':
    unittest.main()
