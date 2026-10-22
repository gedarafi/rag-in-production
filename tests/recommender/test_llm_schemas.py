import json
import unittest

from src.recommender.llm_schemas import (ConceptRole, Decomposition, MentionInterpretation, NumericRange, json_schema,
                                         parse)
from src.recommender.mention_interpreter import validate

MENTION = {'meaning': 'Requested title', 'text': 'Air', 'kind': 'anime', 'role': 'subject', 'query': 'Air'}


class SchemaTests(unittest.TestCase):
    def test_schemas_are_self_contained_and_closed(self):
        schema = json_schema(MentionInterpretation)
        text = json.dumps(schema)
        self.assertNotIn('$ref', text)
        self.assertNotIn('title', schema)
        self.assertFalse(schema['additionalProperties'])
        item = schema['properties']['mentions']['items']
        self.assertEqual(item['properties']['meaning'], {'type': 'string', 'maxLength': 180})
        self.assertEqual(json_schema(ConceptRole)['properties']['role']['enum'],
                         ['required', 'excluded', 'preferred', 'irrelevant'])

    def test_schemas_stay_exactly_as_the_steps_were_tuned_with(self):
        # Ollama's constrained decoding changes with the schema, down to key and `required` order: with
        # `meaning` before `text`, llama3.2:3b stopped quoting the request. These are the hand-written
        # schemas the pipeline used before the models replaced them.
        strings = {'type': 'array', 'items': {'type': 'string'}}
        expected = {
            MentionInterpretation: {
                'type': 'object', 'additionalProperties': False,
                'required': ['intent', 'use_previous_context', 'mentions'], 'properties': {
                    'intent': {'enum': ['recommendation', 'factual', 'other']},
                    'use_previous_context': {'type': 'boolean'},
                    'mentions': {'type': 'array', 'maxItems': 16, 'items': {
                        'type': 'object', 'additionalProperties': False,
                        'required': ['meaning', 'text', 'kind', 'role', 'query'],
                        'properties': {'text': {'type': 'string'}, 'query': {'type': 'string'},
                                       'kind': {'enum': ['anime', 'character', 'genre', 'tag', 'description', 'other']},
                                       'role': {'enum': ['subject', 'reference', 'required', 'excluded', 'preferred',
                                                         'ambiguous']},
                                       'meaning': {'type': 'string', 'maxLength': 180}}}}}},
            Decomposition: {
                'type': 'object', 'additionalProperties': False,
                'required': ['intent', 'use_previous_context', 'entity_queries', 'required', 'excluded', 'preferred'],
                'properties': {'intent': {'enum': ['recommendation', 'factual', 'other']},
                               'use_previous_context': {'type': 'boolean'},
                               **{key: {**strings, 'maxItems': 8}
                                  for key in ('entity_queries', 'required', 'excluded', 'preferred')}}},
            ConceptRole: {'type': 'object', 'additionalProperties': False, 'required': ['role'],
                          'properties': {'role': {'enum': ['required', 'excluded', 'preferred', 'irrelevant']}}},
            NumericRange: {'type': 'object', 'required': ['min', 'max'],
                           'properties': {'min': {'type': ['number', 'null']}, 'max': {'type': ['number', 'null']}}},
        }

        def orders(node):
            if isinstance(node, dict):
                own = [list(node['properties']), node.get('required')] if 'properties' in node else []
                return own + [order for value in node.values() for order in orders(value)]
            return [order for value in node for order in orders(value)] if isinstance(node, list) else []

        for model, schema in expected.items():
            with self.subTest(model=model.__name__):
                self.assertEqual(json_schema(model), schema)
                self.assertEqual(orders(json_schema(model)), orders(schema))

    def test_replies_keep_their_json_types(self):
        decomposition = {'intent': 'factual', 'use_previous_context': False, 'entity_queries': ['Dune'],
                         'required': [], 'excluded': [], 'preferred': []}
        parse(Decomposition, decomposition)
        for change in ({'use_previous_context': 'false'}, {'intent': 'chat'}, {'required': ['a'] * 9},
                       {'extra': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse(Decomposition, {**decomposition, **change})

    def test_errors_are_short_enough_to_send_back(self):
        with self.assertRaises(ValueError) as raised:
            parse(MentionInterpretation, {'intent': 'factual', 'use_previous_context': False,
                                          'mentions': [{**MENTION, 'kind': 'show'}]})
        message = str(raised.exception)
        self.assertIn('mentions.0.kind', message)
        self.assertNotIn('https://', message)

    def test_mentions_are_checked_before_their_meaning(self):
        with self.assertRaisesRegex(ValueError, 'mentions.0.role'):
            validate({'intent': 'factual', 'use_previous_context': False, 'mentions': [{**MENTION, 'role': 'x'}]},
                     'What is Air about?', [])
        decomposition, kept = validate({'intent': 'factual', 'use_previous_context': False, 'mentions': [MENTION]},
                                       'What is Air about?', [])
        self.assertEqual(decomposition['entity_queries'], ['Air'])

    def test_numeric_bounds_must_be_finite_numbers_or_null(self):
        parse(NumericRange, {'min': 3, 'max': None})
        for bound in (float('inf'), True, '3'):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                parse(NumericRange, {'min': bound, 'max': None})


if __name__ == '__main__':
    unittest.main()
