import json
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from src.recommender.answer_arithmetic import evaluate_quantity, render_quantities
from src.recommender.providers import LocalModels


class AnswerArithmeticTests(unittest.TestCase):
    def test_source_units_prevent_count_time_mixups(self):
        measurements = {'q0': {'value': 10, 'units': {'item': 1}},
                        'q1': {'value': 20, 'units': {'minute': 1, 'item': -1}},
                        'q2': {'value': 5, 'units': {'item': 1}}}
        text, _ = render_quantities([
            {'label': 'Combined count', 'expression': 'q0+q2', 'unit': 'minutes', 'decimals': 0},
            {'label': 'Total time', 'expression': '(q0+q2)*q1', 'unit': 'hours', 'decimals': 1}], measurements)
        self.assertIn('Combined count: 15 item.', text)
        self.assertIn('Total time: 5 hours.', text)
        for expression in ('q0+q1', '12', 'unknown', '__import__("os")'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                evaluate_quantity(expression, measurements)
    def test_general_expressions_and_dependencies(self):
        self.assertEqual(evaluate_quantity('(10 + 20 + 30) / 3', {})[0], Decimal('20'))
        text, checks = render_quantities([
            {'label': 'First', 'expression': '10*20', 'unit': 'auto', 'decimals': 0},
            {'label': 'Second', 'expression': '5*30', 'unit': 'auto', 'decimals': 0},
            {'label': 'Combined', 'expression': 'c0+c1', 'unit': 'auto', 'decimals': 0}], {})
        self.assertEqual(text, 'First: 200.\nSecond: 150.\nCombined: 350.')
        self.assertEqual(checks[-1]['value'], '350')
        self.assertEqual(render_quantities([{'label': 'Average', 'expression': '10/3', 'unit': 'auto', 'decimals': 2}], {})[0], 'Average: 3.33.')

    def test_code_and_unbounded_operations_cannot_execute(self):
        for expression in ('__import__("os")', '1.__class__', '[1][0]', '2**100000', '1/0', 'True', 'c1+1', '1e100'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                evaluate_quantity(expression, {})

    def test_invalid_quantity_items_are_rejected(self):
        for items in ([], [{'expression': '2+3'}],
                      [{'label': 'Bad', 'expression': '2+3', 'unit': 'auto', 'decimals': 50}]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                render_quantities(items, {})


class GroundedReplyTests(unittest.TestCase):
    def setUp(self):
        self.models = LocalModels('unused', 'http://example.test', 'test')

    def call_reply(self, result, subjects, measurements=None):
        response = Mock()
        response.json.return_value = {'message': {'content': json.dumps(result)}}
        with patch('src.recommender.providers.requests.post', return_value=response) as post:
            answer = self.models.agent_fact_reply('Source data', .5, 500, subjects, measurements)
        return answer, post.call_args.kwargs['json']

    def test_only_prose_is_displayed_and_calculation_is_evaluated(self):
        result = {'kind': 'quantities', 'supporting_facts': [{'subject': 'item:a', 'fact': '10 items'},
                                       {'subject': 'item:b', 'fact': '20 items'}],
                  'items': [{'label': 'Combined count', 'expression': 'q0+q1', 'unit': 'auto', 'decimals': 0}]}
        answer, request = self.call_reply(result, ['item:a', 'item:b'], {
            'q0': {'value': 10, 'units': {'item': 1}}, 'q1': {'value': 20, 'units': {'item': 1}}})
        self.assertEqual(answer, 'Combined count: 30 item.')
        self.assertIn('format', request)
        self.assertEqual(self.models.last_fact_grounding['calculations'][0]['value'], '30')

    def test_missing_subject_or_invented_evidence_is_rejected(self):
        result = {'kind': 'prose', 'supporting_facts': [{'subject': 'item:a', 'fact': 'An unsupported fact.'}],
                  'answer': 'A guessed answer.'}
        for subjects in ([], ['item:a', 'item:b']):
            with self.subTest(subjects=subjects), self.assertRaisesRegex(RuntimeError, 'invalid grounded response'):
                self.call_reply(result, subjects)

    def test_comparison_uses_model_prose_without_calculations(self):
        answer, _ = self.call_reply({'kind': 'prose', 'supporting_facts': [{'subject': 'a', 'fact': 'Fantasy setting'}],
                                     'answer': 'Both stories feature fantasy settings.'}, ['a'])
        self.assertEqual(answer, 'Both stories feature fantasy settings.')

    def test_no_sources_requires_clarification(self):
        answer, _ = self.call_reply({'kind': 'prose', 'supporting_facts': [], 'answer': 'Which work do you mean?'}, [])
        self.assertEqual(answer, 'Which work do you mean?')
        with self.assertRaisesRegex(RuntimeError, 'invalid grounded response'):
            self.call_reply({'kind': 'prose', 'supporting_facts': [], 'answer': 'It has 12 episodes.'}, [])
