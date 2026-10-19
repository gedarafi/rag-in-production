import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from unittest.mock import patch

from tests.recommender import test_entity_agent as fixtures
from src.recommender.entity_agent import EntityAgent, independent_recommendation
from src.recommender.tool_plan import resolve_references
from src.recommender.evaluation_runtime import span_summary
from opentelemetry import trace

from src.tracing import capture_spans


def step(tool, **arguments):
    return {'tool': tool, 'arguments': arguments}


class ToolPlanTests(unittest.TestCase):
    def setUp(self):
        fixtures.EntityAgentTests.setUp(self)
        self.agent.answer = EntityAgent.answer.__get__(self.agent)
        self.calls = []
        self.plan = {'scope': 'in_domain', 'direct_response': '', 'use_previous_context': False,
                     'goal': 'Answer the question from sources.', 'clarification': '', 'steps': []}
        self.reply = 'A natural sourced answer. [anime-1.md]'
        self.final_prompts = []
        def plan(prompt, schema):
            self.calls.append('plan')
            self.prompts.append((prompt, schema))
            return self.plan if isinstance(self.plan, str) else json.dumps(self.plan)
        def finish(prompt, temperature, max_tokens):
            self.calls.append('final')
            self.final_prompts.append(prompt)
            return self.reply
        self.models.plan_tools = plan
        self.models.agent_reply = finish

    def ask(self, question, history=None):
        result = self.agent.answer(question, history or [])
        self.assertEqual(self.calls, ['plan', 'final'])
        self.assertEqual(self.agent.last_usage['llm_calls'], 2)
        return result

    def test_simple_fact_two_calls_with_no_extra_retrieval(self):
        self.reply = 'Re:Zero has 25 episodes. [anime-1.md]'
        self.assertIn('25 episodes', self.ask('How long is Re:Zero?'))
        self.assertEqual(self.agent.last_usage['executed_tools'], 0)
        self.assertIn('"episodes": 25.0', self.final_prompts[0])

    def test_separately_mentioned_prefix_title_is_not_swallowed_by_sequel(self):
        records = [{'id': 'base', 'aliases': ['Example']},
                   {'id': 'sequel', 'aliases': ['Example Second Season']}]
        self.assertEqual(self.agent.explicit_titles(records, 'Compare Example and Example Second Season'),
                         ['base', 'sequel'])
        self.assertEqual(self.agent.explicit_titles(records, 'How long is Example Second Season?'), ['sequel'])
        self.assertEqual(self.agent.explicit_titles(records, 'Example Second Season versus Example'),
                         ['sequel', 'base'])
        self.assertEqual(self.agent.explicit_titles(records, 'Example and Example'), ['base'])
        records.append({'id': 'ambiguous', 'aliases': ['Example']})
        self.assertEqual(self.agent.explicit_titles(records, 'How long is Example?'), [])

    def test_comparison_keeps_all_subject_facts_after_tools_and_context_trimming(self):
        with closing(sqlite3.connect(self.catalog.database)) as db:
            db.execute("UPDATE anime SET data=json_set(data,'$.episodes',12,'$.duration',30) WHERE id='2'")
            db.commit()
        self.agent.settings = replace(self.agent.settings, context_chars=500)
        self.plan['steps'] = [step('get_anime', id='1'), step('get_anime', id='2')]
        self.ask('What are the similarities between Re:Zero and Other World?')
        self.assertEqual({row['id'] for row in self.agent.last_usage['selected_context']['anime']}, {'1', '2'})
        prompt = self.final_prompts[-1]
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"history":'):])
        self.assertEqual({row['id'] for row in payload['primary_entity_facts']}, {'1', '2'})
        self.assertIn('older_bundle_not_in_window', prompt)
        self.assertEqual({row['title'] for row in payload['primary_entity_facts']}, {'Re:Zero', 'Other World'})
        facts = {row['id']: row for row in payload['primary_entity_facts']}
        self.assertIn('Subaru', facts['1']['description'])
        self.assertIn('Another ordinary boy', facts['2']['description'])
        self.assertEqual(facts['1']['episodes'], 25)
        self.assertEqual(facts['2']['episodes'], 12)
        remembered = next(reversed(self.agent.memory.values()))
        self.assertEqual(set(remembered['active_anime_ids']), {'1', '2'})

    def test_resolved_second_subject_is_retained_with_explicit_first_subject(self):
        self.plan['steps'] = [step('resolve_entity', kind='anime', query='Other World'),
                              step('get_anime', id='$0.matches.0.id')]
        self.ask('What do Re:Zero and OW have in common?')
        remembered = next(reversed(self.agent.memory.values()))
        self.assertEqual(set(remembered['active_anime_ids']), {'1', '2'})
        self.assertIn('Subaru is transported', self.final_prompts[-1])
        self.assertIn('Another ordinary boy is transported', self.final_prompts[-1])
        self.assertEqual(self.agent.last_usage['llm_calls'], 2)

    def test_multi_title_followup_rehydrates_all_subjects(self):
        history = [{'role': 'user', 'content': 'Compare Re:Zero and Other World'},
                   {'role': 'assistant', 'content': 'An earlier answer.'}]
        self.plan['use_previous_context'] = True
        self.ask('What do they have in common?', history)
        self.assertEqual({row['id'] for row in self.agent.last_usage['selected_context']['anime']}, {'1', '2'})
        self.assertIn('Subaru is transported to another world', self.final_prompts[-1])
        self.assertIn('Another ordinary boy is transported', self.final_prompts[-1])

    def test_semantic_feature_search_feeds_the_same_plan_without_rebinding_references(self):
        self.plan['steps'] = [step('search_features', kind='tag', query='protagonist moral decline', limit=3),
                              step('resolve_entity', kind='genre', query='Fantasy'),
                              step('recommend_anime', genre_ids=['$1.matches.0.id'],
                                   tag_ids=['$0.matches.*.id'], tag_match='any', query='MC turns bad')]
        with patch.object(self.agent, 'feature_candidates', side_effect=[[], [
            {'kind': 'tag', 'id': 'time loop', 'name': 'Time Loop', 'similarity': .9}]]):
            self.ask('Give me anime where the MC turned bad')
        self.assertEqual(self.agent.last_trace[2]['arguments']['tag_ids'], ['time loop'])
        self.assertEqual(self.agent.last_trace[2]['arguments']['genre_ids'], ['fantasy'])
        self.assertEqual(self.agent.last_usage['executed_tools'], 3)
        self.assertIn('authorizes the spoilers needed', self.final_prompts[-1])
        self.assertIn('does not prove a good-to-evil transition', self.final_prompts[-1])

    def test_semantic_tag_search_returns_only_existing_tags_and_expands_mc(self):
        captured = []
        embed = self.models.embed
        def capture(texts):
            captured.extend(texts)
            return embed(texts)
        self.models.embed = capture
        result = self.agent.execute('search_features', {'query': 'MC turned bad', 'kind': 'tag', 'limit': 3}, {})
        self.assertTrue(result['matches'])
        self.assertTrue(all(r['kind'] == 'tag' and r['id'] == 'time loop' for r in result['matches']))
        self.assertIn('main protagonist turned bad', captured)

    def test_empty_feature_search_cannot_widen_into_unfiltered_recommendations(self):
        self.plan['steps'] = [step('search_features', query='unknown arc'),
                              step('recommend_anime', tag_ids=['$0.matches.*.id'], tag_match='any')]
        with patch.object(self.agent, 'feature_candidates', return_value=[]):
            self.ask('Give me anime where the MC turned bad')
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)
        self.assertIn('matched no items', self.agent.last_trace[-1]['error'])

    def test_missing_link_to_one_semantic_tag_search_is_repaired_locally(self):
        self.plan['steps'] = [step('search_features', query='moral decline'),
                              step('recommend_anime', tag_match='any', query='MC turns bad')]
        with patch.object(self.agent, 'feature_candidates', side_effect=[[], [
            {'kind': 'tag', 'id': 'time loop', 'name': 'Time Loop', 'similarity': .9}]]):
            self.ask('Give me anime where the MC turned bad')
        self.assertEqual(self.agent.last_trace[1]['arguments']['tag_ids'], ['time loop'])
        self.assertIn('any_terms', self.final_prompts[-1])

    def test_alternative_tags_without_a_search_cannot_fall_back_to_popularity(self):
        self.plan['steps'] = [step('recommend_anime', tag_match='any', query='MC turns bad')]
        self.ask('Give me anime where the MC turned bad')
        self.assertEqual(self.agent.last_usage['executed_tools'], 0)
        self.assertIn('require tag IDs', self.agent.last_trace[0]['error'])

    def test_unrelated_request_gets_llm_written_reply_without_tools_or_final_call(self):
        self.plan.update(scope='out_of_domain', direct_response='Slice the apple and simmer it with a little water until soft.')
        with patch.object(self.agent, 'execute', side_effect=AssertionError('must not run tools')), \
             patch.object(self.agent, 'finish', side_effect=AssertionError('must not call final model')), \
             capture_spans() as spans:
            result = self.agent.answer('How to cook an apple?', [])
        self.assertEqual(result, self.plan['direct_response'])
        self.assertEqual(self.calls, ['plan'])
        self.assertEqual(self.agent.last_usage['llm_calls'], 1)
        self.assertEqual(self.agent.last_usage['executed_tools'], 0)
        self.assertEqual(self.agent.last_usage['response_path'], 'planner_direct_response')
        (decision,) = [e for span in spans for e in span.events if e.name == 'scope_decision']
        self.assertEqual(decision.attributes['scope'], 'out_of_domain')

    def test_uncertain_request_keeps_normal_flow(self):
        self.plan.update(scope='uncertain', clarification='Ask what Apple refers to.')
        self.reply = 'Do you mean a title or something in the previous episode?'
        self.assertEqual(self.ask('Apple?'), self.reply)
        self.assertEqual(self.agent.last_usage['scope'], 'uncertain')

    def test_scope_does_not_override_explicit_anime(self):
        self.plan.update(scope='out_of_domain', direct_response='A mistaken general response.')
        self.ask('What was Re:Zero about?')
        self.assertEqual(self.agent.last_usage['scope'], 'uncertain')
        self.assertIn('returns after death', self.final_prompts[-1])

    def test_scope_does_not_override_independent_recommendations(self):
        self.plan.update(scope='out_of_domain', direct_response='A mistaken general response.')
        self.ask('Recommend some cooking anime')
        self.assertEqual(self.agent.last_usage['scope'], 'uncertain')

    def test_scope_does_not_override_contextual_followup(self):
        self.plan.update(scope='out_of_domain', use_previous_context=True,
                         direct_response='An unrelated cooking answer.')
        self.ask('Why did she cook it?', [{'role': 'user', 'content': 'What is Re:Zero?'}])
        self.assertEqual(self.agent.last_usage['scope'], 'uncertain')
        self.assertTrue(self.agent.last_usage['use_previous_context'])

    def test_contradictory_scope_with_tools_keeps_normal_flow(self):
        self.plan.update(scope='out_of_domain', direct_response='A mistaken general response.',
                         steps=[step('popular', limit=1)])
        self.ask('What is most popular?')
        self.assertEqual(self.agent.last_usage['scope'], 'uncertain')
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)

    def test_empty_or_invalid_scope_response_does_not_shortcut(self):
        for scope, direct in [('out_of_domain', ''), ('out_of_domain', '[anime-1.md]'), ('invalid', 'No.')]:
            with self.subTest(scope=scope, direct=direct):
                self.calls.clear()
                self.plan.update(scope=scope, direct_response=direct)
                self.ask('Apple?')

    def test_direct_reply_clears_old_topic_for_followups(self):
        self.plan.update(scope='out_of_domain', direct_response='Simmer apple slices until soft.')
        history = [{'role': 'user', 'content': 'What was Re:Zero about?'},
                   {'role': 'assistant', 'content': 'A prior anime discussion.'}]
        question = 'How to cook an apple?'
        result = self.agent.answer(question, history)
        history += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': result}]
        _, state, _, _ = self.agent.prepare('Why did she cook it?', history)
        self.assertEqual(state['active_anime_ids'], [])

    def test_each_stage_and_tool_reports_elapsed_seconds(self):
        self.plan['steps'] = [step('get_anime', id='1')]
        with patch('src.recommender.entity_agent.perf_counter', side_effect=range(20)):
            self.ask('What is Re:Zero?')
        timings = self.agent.last_usage['timings_seconds']
        self.assertEqual(timings, {name: 1 for name in (
            'prepare_context', 'retrieve_guides', 'feature_candidates',
            'planning', 'tools', 'generate_answer')})
        self.assertEqual(self.agent.last_trace[0]['elapsed_seconds'], 1)

    def test_failed_planning_retains_elapsed_time(self):
        self.models.plan_tools = lambda *args: (_ for _ in ()).throw(RuntimeError('offline'))
        with patch('src.recommender.entity_agent.perf_counter', side_effect=range(20)):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                self.agent.answer('What is Re:Zero?', [])
        self.assertEqual(self.agent.last_usage['timings_seconds']['planning'], 1)

    def test_tools_run_in_their_own_spans_before_final_generation(self):
        self.plan['steps'] = [step('get_anime', id='1')]
        original_execute = self.agent.execute
        def execute(*args):
            # The tool runs inside its TOOL span, after the plan was recorded.
            self.assertEqual(trace.get_current_span().name, 'get_anime')
            self.assertEqual(self.agent.last_usage['planned_tools'], 1)
            return original_execute(*args)
        def reply(*args):
            tool_done = [span_summary(s) for s in spans if span_summary(s)['stage'] == 'tool']
            self.assertEqual([(t['tool'], t['status']) for t in tool_done], [('get_anime', 'completed')])
            self.assertIn('elapsed_seconds', tool_done[0])
            return 'A natural sourced answer.'
        self.agent.execute = execute
        self.models.agent_reply = reply
        with capture_spans() as spans:
            self.agent.answer('What is Re:Zero?', [])
        stages = [span_summary(span) for span in spans]
        # The answer is the last stage; the agent span around all of them ends after it.
        self.assertEqual(stages[-2]['stage'], 'generate_answer')
        self.assertEqual(stages[-2]['status'], 'completed')
        self.assertEqual(stages[-1]['kind'], 'agent')

    def test_failed_planning_is_an_error_span_before_raising(self):
        self.models.plan_tools = lambda *args: (_ for _ in ()).throw(RuntimeError('offline'))
        with self.assertRaisesRegex(RuntimeError, 'offline'), capture_spans() as spans:
            self.agent.answer('What is Re:Zero?', [])
        stages = [span_summary(span) for span in spans]
        self.assertEqual(stages[-2]['stage'], 'planning')
        self.assertEqual(stages[-2]['status'], 'error')
        self.assertIn('elapsed_seconds', stages[-2])
        self.assertEqual(stages[-1]['status'], 'error')

    def test_recommendations_keep_aliases_and_matching_features_per_work(self):
        bundle = self.store.anime('1')
        self.assertNotIn('characters', bundle)
        self.assertEqual(bundle['main_characters'][0]['name'], 'Subaru Natsuki')
        recommendation = bundle['recommendations'][0]
        self.assertEqual(recommendation['aliases'], ['Other World'])
        self.assertEqual(recommendation['shared_genres'], ['Fantasy'])
        self.assertEqual(recommendation['shared_tags'], [])

    def test_unsupported_alias_and_cross_work_character_are_rejected(self):
        evidence = [{'tool': 'get_anime', 'result': self.store.anime('1')}]
        wrong = ('Try "Other World" (also known as "Memories of the Alhambra"). '
                 "It resembles Subaru's ability in Other World.")
        issues = self.agent.identity_issues(wrong, evidence)
        self.assertTrue(any('alias' in issue for issue in issues))
        self.assertTrue(any('cast' in issue for issue in issues))
        self.assertEqual(self.agent.identity_issues(
            'Re:Zero (also known as "ReZero") features Subaru Natsuki. '
            'Try Other World for its shared Fantasy genre.', evidence), [])
        self.assertEqual(self.agent.identity_issues(
            'Subaru is not in Other World.', evidence), [])

    def test_identity_checks_keep_case_insensitive_attribution(self):
        evidence = [{'tool': 'get_anime', 'result': self.store.anime('1')}]
        self.assertTrue(self.agent.identity_issues(
            "SUBARU'S ability in OTHER WORLD is unusual.", evidence))

    def test_planner_receives_evidence_availability_without_full_bundles(self):
        self.ask('What was Re:Zero about?')
        prompt = self.prompts[0][0]
        data, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('\n{') + 1:])
        summary = data['evidence'][0]['result']
        self.assertEqual(summary['id'], '1')
        self.assertIn('description', summary['available_fields'])
        self.assertEqual(summary['recommendations_count'], 1)
        self.assertNotIn('description', summary)
        self.assertNotIn('retrieved_guides', data)
        self.assertNotIn('current_message_entity_ids', data)
        self.assertIn('returns after death', self.final_prompts[0])

    def test_synopsis_evidence_deduplicates_overview_and_omits_recommendations(self):
        bundle = self.store.anime('1')
        evidence = [{'tool': 'entity_context', 'result': bundle},
                    {'tool': 'get_anime', 'result': bundle}]
        selected = self.agent.answer_evidence('What was Re:Zero about?', evidence)
        self.assertEqual(len(selected), 1)
        self.assertIn('returns after death', selected[0]['result']['description'])
        self.assertNotIn('recommendations', selected[0]['result'])
        # Prompt trimming must not mutate the trusted evidence used for validation.
        self.assertEqual(len(bundle['recommendations']), 1)

    def test_answer_evidence_preserves_distinct_cast_and_profile_pages(self):
        evidence = [{'tool': 'get_anime', 'result': self.store.anime('1')},
                    {'tool': 'get_anime', 'result': self.store.anime('1', section='cast')},
                    {'tool': 'get_character', 'result': self.store.character('17')}]
        selected = self.agent.answer_evidence('Who are the characters?', evidence)
        self.assertEqual(len(selected), 3)
        self.assertIn('recommendations', selected[0]['result'])
        self.assertIn('He returns after death', selected[1]['result']['cast'][0]['description'])
        self.assertIn('protects his friends', selected[2]['result']['description'])

    def test_recommendation_tool_evidence_is_retained_for_synopsis_wording(self):
        evidence = [{'tool': 'recommend_anime', 'result': {'items': [self.store.anime('1')]}}]
        selected = self.agent.answer_evidence('What was that about?', evidence)
        self.assertEqual(selected, evidence)

    def test_summarizing_recommendations_keeps_nested_matches(self):
        evidence = [{'tool': 'get_anime', 'result': self.store.anime('1')}]
        selected = self.agent.answer_evidence('Summarize recommendations for Re:Zero', evidence)
        self.assertEqual(selected, evidence)

    def test_identity_error_gets_one_correction_and_tracks_actual_calls(self):
        replies = iter(['Other World (also known as "An invented title").',
                        'Try Other World for its shared Fantasy genre. [anime-2.md]'])
        self.models.agent_reply = lambda *args: next(replies)
        result = self.agent.answer('Recommend anime like Re:Zero', [])
        self.assertIn('shared Fantasy', result)
        self.assertEqual(self.agent.last_usage['llm_calls'], 3)
        self.assertTrue(self.agent.last_usage['identity_corrections'])

    def test_repeated_identity_error_is_not_returned(self):
        self.models.agent_reply = lambda *args: 'Other World (also known as "An invented title").'
        with self.assertRaisesRegex(RuntimeError, 'mixed anime identities'):
            self.agent.answer('Recommend anime like Re:Zero', [])

    def test_unknown_title_resolved_and_loaded_in_single_plan(self):
        self.plan['steps'] = [step('resolve_entity', kind='anime', query='ReZro'),
                              step('get_anime', id='$0.matches.0.id')]
        self.ask('What is ReZro?')
        self.assertEqual(self.agent.last_usage['executed_tools'], 2)
        self.assertEqual(self.agent.last_trace[1]['arguments']['id'], '1')
        self.assertIn('"episodes": 25.0', self.final_prompts[0])

    def test_layered_main_character_recommendation_two_calls(self):
        self.plan['steps'] = [step('get_anime', id='1'),
                              step('get_character', id='$0.main_characters.0.id'),
                              step('recommend_by_character', character_ids=['$1.id'], query='ordinary transported boy')]
        self.reply = 'Try Other World for its similar main character. [anime-2.md]'
        self.assertIn('Other World', self.ask('Anime whose main character resembles Re:Zero?'))
        self.assertEqual(self.agent.last_usage['executed_tools'], 3)
        self.assertIn('"matching_character"', self.final_prompts[0])
        self.assertNotIn('"title": "Unrelated Show"', self.final_prompts[0])

    def test_out_of_order_dictionary_reference_is_bound_by_result_type(self):
        self.plan['steps'] = [step('get_character', id='$0.matches.0.id'),
                              step('resolve_entity', kind='character', query='Subaru', anime_ids=['1']),
                              step('recommend_by_character', character_ids=['$0.id'], query='transported boy')]
        self.ask('Recommend anime with a main character like Re:Zero')
        self.assertEqual([s['step'] for s in self.agent.last_trace], [1, 0, 2])
        self.assertEqual(self.agent.last_usage['executed_tools'], 3)
        self.assertIn('"matching_character"', self.final_prompts[0])

    def test_forward_dependency_is_scheduled_before_its_consumer(self):
        self.plan['steps'] = [step('get_anime', id='$1.matches.0.id'),
                              step('resolve_entity', kind='anime', query='ReZro')]
        self.ask('What is ReZro?')
        self.assertEqual([s['step'] for s in self.agent.last_trace], [1, 0])
        self.assertEqual(self.agent.last_usage['executed_tools'], 2)

    def test_genre_and_tag_dependencies(self):
        self.plan['steps'] = [step('resolve_entity', kind='genre', query='SciFi'),
                              step('resolve_entity', kind='tag', query='Time Lop'),
                              step('popular', genre_ids=['$0.matches.0.id'], tag_ids=['$1.matches.0.id'])]
        self.ask('Most popular SciFi anime with Time Lop?')
        self.assertEqual(self.agent.last_trace[-1]['arguments']['genre_ids'], ['sci fi'])
        self.assertEqual(self.agent.last_trace[-1]['arguments']['tag_ids'], ['time loop'])

    def test_dictionary_id_shorthand_is_normalized(self):
        self.plan['steps'] = [step('resolve_entity', kind='tag', query='Time Loop'),
                              step('recommend_anime', tag_ids=['$0.id'])]
        self.ask('Recommend anime with Time Loop')
        self.assertEqual(self.agent.last_trace[-1]['arguments']['tag_ids'], ['time loop'])

    def test_known_filter_type_is_corrected_without_another_llm_call(self):
        self.plan['steps'] = [step('resolve_entity', kind='tag', query='Time Loop'),
                              step('recommend_anime', genre_ids=['$0.id'], query='time travel')]
        self.ask('Recommend anime with Time Loop')
        self.assertEqual(self.agent.last_usage['executed_tools'], 2)
        self.assertIn('Re:Zero', self.final_prompts[-1])

    def test_no_plan_repair_calls_for_malformed_json(self):
        self.plan = '{bad json'
        self.reply = 'Which series do you mean?'
        self.assertEqual(self.ask('How many episodes?'), self.reply)
        self.assertIn('invalid_plan', self.final_prompts[0])

    def test_failed_dependency_does_not_block_independent_tool(self):
        self.plan['steps'] = [step('get_anime', id='999'),
                              step('get_character', id='$0.main_characters.0.id'), step('popular', limit=1)]
        self.ask('Compare this unknown anime with the most popular one')
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)
        self.assertIn('required earlier step failed', self.final_prompts[0])
        self.assertIn('Unrelated Show', self.final_prompts[0])

    def test_repeated_functions_are_deduplicated_without_replanning(self):
        self.plan['steps'] = [step('popular', limit=2), step('popular', limit=2)]
        self.ask('What is most popular?')
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)
        self.assertTrue(self.agent.last_trace[-1]['cached'])

    def test_invalid_arguments_are_not_executed(self):
        self.plan['steps'] = [step('popular', limit=True)]
        self.ask('What is most popular?')
        self.assertEqual(self.agent.last_usage['executed_tools'], 0)
        self.assertIn('Invalid integer', self.final_prompts[0])

    def test_clarification_is_written_by_final_llm(self):
        self.plan['clarification'] = 'No anime or character has been identified.'
        self.reply = 'Which anime were you thinking of?'
        self.assertEqual(self.ask('Who is the main character?'), self.reply)
        self.assertIn('No anime or character has been identified', self.final_prompts[0])

    def test_references_cannot_address_future_steps(self):
        with self.assertRaises(ValueError):
            resolve_references('$1.matches.0.id', [{'matches': []}])

    def test_wildcard_reference_flattens_id_lists(self):
        results = [{'main_characters': [{'id': '17'}, {'id': '18'}]}]
        self.assertEqual(resolve_references(['$0.main_characters.*.id'], results), ['17', '18'])

    def test_follow_up_keeps_reference_anime(self):
        self.plan['steps'] = [step('recommend_anime', seed_anime_ids=['1'])]
        self.reply = 'Try Other World. [anime-2.md]'
        question = 'Recommend anime like Re:Zero'
        self.ask(question)
        self.calls.clear()
        self.plan['use_previous_context'] = True
        self.plan['steps'] = []
        self.reply = 'There are 25 episodes. [anime-1.md]'
        self.ask('How many episodes are there?', [{'role': 'user', 'content': question},
                                                 {'role': 'assistant', 'content': 'Try Other World. [anime-2.md]'}])
        self.assertIn('"reference_anime": ["Re:Zero"]', self.final_prompts[-1])

    def test_router_can_use_previous_context_for_cast(self):
        self.plan['use_previous_context'] = True
        self.plan['steps'] = [step('get_anime', id='1', section='cast', limit=50)]
        self.reply = 'The main character is Subaru Natsuki. [anime-1.md]'
        self.ask('Who are the characters?', [{'role': 'user', 'content': 'What is Re:Zero?'},
                                            {'role': 'assistant', 'content': 'An earlier synopsis.'}])
        self.assertTrue(self.agent.last_usage['use_previous_context'])
        self.assertIn('Subaru Natsuki', self.final_prompts[-1])
        self.assertIn('An ordinary transported boy.', self.final_prompts[-1])
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)

    def test_cast_biographies_are_scoped_and_overview_stays_compact(self):
        cast = self.store.anime('1', section='cast')
        self.assertEqual(cast['cast'][0]['name'], 'Subaru Natsuki')
        self.assertIn('He returns after death', cast['cast'][0]['description'])
        self.assertNotIn('Another Hero', json.dumps(cast))
        self.assertNotIn('description', self.store.anime('1')['main_characters'][0])
        self.assertEqual(self.store.anime('1', section='cast', offset=1)['cast'], [])

    def test_router_can_discard_previous_context_and_history(self):
        self.plan['use_previous_context'] = False
        self.ask('Hello', [{'role': 'user', 'content': 'What is Re:Zero?'},
                           {'role': 'assistant', 'content': 'A stale assistant answer.'}])
        self.assertIn('"history": []', self.final_prompts[-1])
        self.assertNotIn('A stale assistant answer', self.final_prompts[-1])
        self.assertNotIn('"episodes": 25.0', self.final_prompts[-1])
        self.assertEqual(self.agent.last_usage['selected_context']['anime'], [])

    def test_prior_id_selection_repairs_contradictory_context_flag(self):
        self.plan['steps'] = [step('get_anime', id='1')]
        self.ask('How long is it?', [{'role': 'user', 'content': 'What is Re:Zero?'}])
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)
        self.assertTrue(self.agent.last_usage['use_previous_context'])
        self.assertIn('context_repair', self.agent.last_usage)
        self.assertIn('"episodes": 25.0', self.final_prompts[-1])

    def test_followup_keeps_group_when_one_prior_id_is_reloaded(self):
        self.plan['steps'] = [step('get_anime', id='1')]
        self.ask('How many episodes is it?', [
            {'role': 'user', 'content': 'Compare Re:Zero and Other World'},
            {'role': 'assistant', 'content': 'They have 999 episodes combined.'}])
        remembered = next(reversed(self.agent.memory.values()))
        self.assertEqual(set(remembered['active_anime_ids']), {'1', '2'})
        self.assertIn('Previous assistant replies are unverified', self.final_prompts[-1])
        self.assertIn('"duration": 24.0', self.final_prompts[-1])
        self.assertNotIn('"recommendation_links"', self.final_prompts[-1])

    def test_referential_followup_keeps_sources_even_if_planner_requests_missing_title(self):
        self.plan['steps'] = [step('get_anime', id='title')]
        self.plan['clarification'] = 'Please provide the anime title.'
        self.ask('How many episodes it is?', [
            {'role': 'user', 'content': 'Compare Re:Zero and Other World'},
            {'role': 'assistant', 'content': 'An incorrect earlier answer.'}])
        self.assertTrue(self.agent.last_usage['use_previous_context'])
        self.assertEqual({row['id'] for row in self.agent.last_usage['selected_context']['anime']}, {'1', '2'})
        self.assertIn('"episodes": 25.0', self.final_prompts[-1])

    def test_fact_response_uses_general_grounding_interface_with_all_subjects(self):
        calls = []
        def fact_reply(prompt, temperature, max_tokens, subjects, measurements, **kwargs):
            calls.append(subjects)
            self.models.last_fact_grounding = {'supporting_facts': [], 'check': 'A source-based comparison.'}
            return 'Both stories involve transportation to another world.'
        self.models.agent_fact_reply = fact_reply
        text = self.agent.answer('What do Re:Zero and Other World have in common?', [])
        self.assertIn('Both stories', text)
        self.assertEqual(set(calls[0]), {'anime:1', 'anime:2'})
        self.assertEqual(self.agent.last_usage['llm_calls'], 2)
        self.assertIn('answer_grounding', self.agent.last_usage)

    def test_invalid_grounded_response_gets_one_counted_correction(self):
        calls = []
        def fact_reply(prompt, *_args, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                self.models.last_fact_validation_error = 'Ambiguous literal: use a measurement variable instead'
                raise RuntimeError('The model returned an invalid grounded response. Please retry.')
            self.models.last_fact_validation_error = ''
            return 'Re:Zero has 25 episodes.'
        self.models.agent_fact_reply = fact_reply
        self.assertIn('25 episodes', self.agent.answer('How many episodes does Re:Zero have?', []))
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.agent.last_usage['llm_calls'], 3)
        self.assertEqual(len(self.agent.last_usage['grounding_corrections']), 1)

    def test_rhetorical_comparison_is_not_forced_into_arithmetic(self):
        modes = []
        def fact_reply(prompt, *_args, **kwargs):
            modes.append(bool(kwargs.get('require_quantities')))
            return 'Both stories involve transportation to another world.'
        self.models.agent_fact_reply = fact_reply
        self.agent.answer('How much do Re:Zero and Other World have in common?', [])
        self.assertEqual(modes, [False])

    def test_discarding_history_keeps_explicit_current_entity(self):
        self.ask('What is Re:Zero?', [{'role': 'user', 'content': 'What is Unrelated Show?'}])
        self.assertIn('"episodes": 25.0', self.final_prompts[-1])
        self.assertNotIn('A supporting character has similar traits', self.final_prompts[-1])

    def test_context_choice_is_not_coerced_from_a_string(self):
        self.plan['use_previous_context'] = 'false'
        self.ask('Hello')
        self.assertIn('must be a boolean', self.final_prompts[-1])

    def test_full_title_dominates_nested_short_alias(self):
        records = [{'id': '1', 'aliases': ['Summer Pockets']}, {'id': '2', 'aliases': ['_summer']}]
        self.assertEqual(self.agent.explicit_title(records, 'What is Summer Pockets?'), '1')
        records = [{'id': '1', 'aliases': ['Re:Zero']}, {'id': '2', 'aliases': ['Other World']}]
        self.assertIsNone(self.agent.explicit_title(records, 'Compare Re:Zero and Other World'))

    def test_role_word_cannot_resolve_to_arbitrary_character(self):
        self.plan['use_previous_context'] = True
        self.plan['steps'] = [step('resolve_entity', kind='character', query='characters', anime_ids=['1'])]
        self.ask('Who are the characters?', [{'role': 'user', 'content': 'What is Re:Zero?'}])
        self.assertEqual(self.agent.last_usage['executed_tools'], 0)
        self.assertIn('cast role is not a person name', self.final_prompts[-1])

    def test_generic_preferences_are_not_automatic_title_seeds(self):
        self.ask('Recommend me some sad anime')
        self.assertEqual(self.agent.last_usage['selected_context']['anime'], [])
        self.assertIn('semantic_feature_candidates', self.prompts[0][0])

    def test_short_mood_is_not_a_fuzzy_feature_name(self):
        with closing(sqlite3.connect(self.catalog.database)) as db:
            db.execute('INSERT INTO entity_dictionary VALUES (?,?,?,?)',
                       ('tag', 'sadism', 'Sadism', '["Sadism"]'))
            db.execute('INSERT INTO entity_keys VALUES (?,?,?)', ('sadism', 'tag', 'sadism'))
            db.commit()
        self.assertEqual(self.store.lookup('tag', 'sad')['matches'], [])
        self.assertEqual(self.store.lookup('tag', 'sadism')['matches'][0]['id'], 'sadism')
        self.assertEqual(self.store.lookup('tag', 'Time Lop')['matches'][0]['id'], 'time loop')

    def test_feature_lookup_uses_exact_dictionary_type(self):
        self.plan['steps'] = [step('resolve_entity', kind='genre', query='Time Loop'),
                              step('recommend_anime', genre_ids=['$0.matches.0.id'])]
        self.ask('Recommend time loop anime')
        self.assertEqual(self.agent.last_trace[0]['result']['kind'], 'tag')
        self.assertIn('Subaru', self.final_prompts[-1])

    def test_display_names_are_normalized_to_known_feature_ids(self):
        self.plan['steps'] = [step('recommend_anime', genre_ids=['Fantasy'], tag_ids=['Time Loop'])]
        self.ask('Recommend time loop anime')
        self.assertEqual(self.agent.last_usage['executed_tools'], 1)
        self.assertIn('"preference": "Recommend time loop anime"', self.final_prompts[-1])

    def test_failed_recommendations_cannot_be_replaced_by_memory_based_titles(self):
        self.plan['steps'] = [step('resolve_entity', kind='genre', query='nonexistent'),
                              step('recommend_anime', genre_ids=['$0.matches.0.id'])]
        self.reply = '1. Other World is a great choice.\n2. Unrelated Show is emotional.'
        with self.assertRaisesRegex(RuntimeError, 'No supported recommendations'):
            self.agent.answer('Recommend me some sad anime', [])
        self.assertEqual(self.calls, ['plan', 'final'])
        self.assertIn('"recommendation_evidence_missing": true', self.final_prompts[-1])
        self.assertNotIn('"tool": "resolve_entity"', self.final_prompts[-1])
        self.assertIn('No matching entity', self.final_prompts[-1])

    def test_missing_recommendations_can_be_explained_without_listing_titles(self):
        self.reply = 'The library search could not find supported recommendations. Could you try another preference?'
        self.assertEqual(self.ask('Recommend me some sad anime'), self.reply)

    def test_failed_feature_names_are_not_presented_as_recommendation_evidence(self):
        evidence = [{'tool': 'resolve_entity', 'arguments': {'kind': 'genre', 'query': 'sadism'},
                     'result': {'status': 'no_matches', 'matches': []}},
                    {'tool': 'recommend_anime', 'result': {'items': [self.store.anime('1')]}}]
        selected = self.agent.answer_evidence('Recommend sad anime', evidence)
        self.assertEqual(len(selected), 1)
        self.assertNotIn('sadism', json.dumps(selected))

    def test_new_preference_search_drops_stale_reference_even_if_planner_keeps_history(self):
        self.plan['use_previous_context'] = True
        self.plan['goal'] = 'Recommend titles like Re:Zero.'
        self.plan['steps'] = [step('recommend_anime', genre_ids=['fantasy'], query='sad anime', limit=2)]
        self.reply = 'Here are some Fantasy titles.'
        self.ask('Give me sad anime recommendations', [
            {'role': 'user', 'content': 'What is Re:Zero?'},
            {'role': 'assistant', 'content': 'A stale synopsis and recommendation.'}])
        self.assertFalse(self.agent.last_usage['use_previous_context'])
        self.assertEqual(self.agent.last_usage['selected_context']['anime'], [])
        final = self.final_prompts[-1]
        self.assertIn('"history": []', final)
        self.assertIn('"reference_anime": []', final)
        self.assertIn('"response_goal": "Give me sad anime recommendations"', final)
        self.assertNotIn('Recommend titles like Re:Zero.', final)
        self.assertNotIn('"context_origin": "previous"', final)
        planner_prompt = self.prompts[0][0]
        planner_data, _ = json.JSONDecoder().raw_decode(planner_prompt[planner_prompt.index('\n{') + 1:])
        self.assertEqual(planner_data['evidence'], [])
        self.assertEqual(planner_data['prior_topic_candidates'], [])
        self.assertEqual(planner_data['history'], [])
        self.assertNotIn('1', planner_data['known_entity_ids']['anime'])
        self.assertNotIn('reference_anime', planner_data)

    def test_standalone_query_cannot_execute_a_stale_seed_from_wrong_plan(self):
        self.plan['use_previous_context'] = True
        self.plan['steps'] = [step('recommend_anime', seed_anime_ids=['1'])]
        self.ask('Give me sad anime recommendations', [
            {'role': 'user', 'content': 'What should I watch after Re:Zero?'},
            {'role': 'assistant', 'content': 'Try Other World.'}])
        self.assertEqual(self.agent.last_usage['executed_tools'], 0)
        self.assertFalse(self.agent.last_usage['use_previous_context'])
        self.assertNotIn('"reference_anime": ["Re:Zero"]', self.final_prompts[-1])
        self.assertIn('Use anime IDs', self.final_prompts[-1])

    def test_history_boundary_recognizes_only_complete_nonreferential_recommendations(self):
        for question in ('Give me sad anime recommendations', 'Recommend five comedy shows',
                         'I want horror anime', 'What should I watch after Steins;Gate?'):
            with self.subTest(question=question):
                self.assertTrue(independent_recommendation(question))
        for question in ('Recommend sad anime like that', 'More anime recommendations',
                         'Suggest anime with similar protagonists', 'Who are the characters?',
                         'Steins;Gate', 'Any recommendations?', 'Recommend something else'):
            with self.subTest(question=question):
                self.assertFalse(independent_recommendation(question))

    def test_empty_reference_after_independent_search_does_not_revive_old_topic(self):
        first = 'What should I watch after Re:Zero?'
        history = [{'role': 'user', 'content': first}, {'role': 'assistant', 'content': 'Try Other World.'}]
        self.plan['steps'] = [step('recommend_anime', genre_ids=['fantasy'], query='sad anime')]
        question = 'Give me sad anime recommendations'
        reply = self.ask(question, history)
        history += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}]
        _, state, evidence, _ = self.agent.prepare('Who are the characters?', history)
        self.assertEqual(state['active_anime_ids'], [])
        self.assertEqual(evidence, [])

    def test_explicit_similarity_follow_up_keeps_seed_and_new_preference(self):
        self.plan['use_previous_context'] = True
        self.plan['steps'] = [step('recommend_anime', seed_anime_ids=['1'], query='sad anime', limit=2)]
        self.reply = 'Try Other World.'
        self.ask('Give me sad anime like that', [{'role': 'user', 'content': 'What is Re:Zero?'}])
        self.assertTrue(self.agent.last_usage['use_previous_context'])
        self.assertIn('"reference_anime": ["Re:Zero"]', self.final_prompts[-1])
        self.assertEqual(self.agent.last_trace[0]['arguments']['query'], 'sad anime')

    def test_title_only_clarification_reply_keeps_pending_cast_question(self):
        self.plan['use_previous_context'] = True
        self.plan['steps'] = [step('get_anime', id='1', section='cast')]
        self.reply = 'The main character is Subaru Natsuki.'
        self.ask('Re:Zero', [{'role': 'user', 'content': 'Who are the characters?'},
                            {'role': 'assistant', 'content': 'Which anime?'}])
        self.assertTrue(self.agent.last_usage['use_previous_context'])
        self.assertIn('Who are the characters?', self.final_prompts[-1])
        self.assertIn('Subaru Natsuki', self.final_prompts[-1])

    def test_very_weak_name_similarity_returns_no_match(self):
        self.assertEqual(self.store.lookup('character', 'characters in Summer Pockets')['matches'], [])

    def test_preference_phrase_is_not_a_fuzzy_anime_title(self):
        self.assertEqual(self.store.lookup('anime', 'sad anime')['matches'], [])


if __name__ == '__main__':
    unittest.main()
