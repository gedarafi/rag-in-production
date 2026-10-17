import unittest

from src.catalog_search import BASE_FIELDS
from src.request_interpreter import FIELD_MEANINGS
from src.semantic_attributes import interpret_attributes, covered_property


class SemanticAttributeTests(unittest.TestCase):
    def test_brevity_policy_is_driven_by_model_semantics(self):
        # No keyword from a language parser is required: the model supplied the meaning.
        result = interpret_attributes(lambda *_: {'role': 'qualitative'}, 'A compact adventure please',
                                      BASE_FIELDS, FIELD_MEANINGS, {'episodes': {'sources': ['compact'], 'interpretations': ['brief']}})
        filters, preferred, notes, sources = result
        self.assertEqual([(f['field'], f['operator'], f['value']) for f in filters], [('episodes', 'lte', '50')])
        self.assertEqual(preferred, [{'field': 'episodes', 'min': 10, 'max': 19}])
        self.assertTrue(notes)
        self.assertEqual(sources, ['compact'])

    def test_calendar_range_uses_model_values_not_phrase_rules(self):
        def model(_prompt, schema):
            if 'operator' in schema['properties']:
                return {'operator': schema['properties']['operator']['enum'][0]}
            return {'min': 1970, 'max': 1979}
        filters, preferred, _, _ = interpret_attributes(
            model,
            'From the seventies please', BASE_FIELDS, FIELD_MEANINGS, {'year': {'sources': ['seventies'], 'interpretations': ['range']}})
        self.assertEqual([(f['operator'], f['value']) for f in filters], [('gte', '1970'), ('lte', '1979')])
        self.assertEqual(preferred, [])

    def test_episode_runtime_stays_separate_from_episode_count(self):
        filters, preferred, _, _ = interpret_attributes(lambda *_: {'role': 'qualitative'},
            'Brief episodes please', BASE_FIELDS, FIELD_MEANINGS, {'duration': {'sources': ['Brief episodes'], 'interpretations': ['brief']}})
        self.assertEqual([(f['field'], f['value']) for f in filters], [('duration', '15')])
        self.assertEqual(preferred, [])

    def test_unrequested_metadata_does_not_trigger_model_calls(self):
        def model(*_):
            self.fail('No measurable property was retrieved')
        self.assertEqual(interpret_attributes(model, 'Emotional adventure', BASE_FIELDS, FIELD_MEANINGS, {}),
                         ([], [], [], []))

    def test_interpreted_source_must_exist_in_user_request(self):
        with self.assertRaisesRegex(ValueError, 'user source'):
            interpret_attributes(lambda *_: {'role': 'qualitative'}, 'Fantasy adventure',
                                 BASE_FIELDS, FIELD_MEANINGS, {'episodes': {'sources': ['compact'], 'interpretations': ['brief']}})
        self.assertTrue(covered_property('Brief Anime', ['Brief'], {'anime'}))
        self.assertFalse(covered_property('Brief anime like Mushishi', ['Brief'], {'anime'}))
