import unittest

from src.recommender.catalog_search import BASE_FIELDS
from src.recommender.request_constraints import validate_request, mentioned_features


class RequestConstraintTests(unittest.TestCase):
    features = [('genre', 'fantasy', 'Fantasy'), ('tag', 'military', 'Military'),
                ('tag', 'school', 'School'), ('genre', 'romance', 'Romance')]

    def request(self, filters, **kwargs):
        return {'intent': 'recommendation', 'use_previous_context': False, 'entity_queries': [],
                'filters': filters, 'soft_query': 'moving and thoughtful', 'limitations': [], **kwargs}

    def constraint(self, field, operator, value, source_text=None, mapping='exact'):
        return dict(field=field, operator=operator, value=value, source_text=source_text or value, mapping=mapping)

    def validate(self, request, question):
        return validate_request(request, question, [], BASE_FIELDS, self.features)

    def test_dictionary_types_are_authoritative_for_all_features_and_polarities(self):
        for name, correct, wrong in [('Fantasy', 'genres', 'tags'), ('Military', 'tags', 'genres')]:
            for op, canonical in [('contains', 'eq'), ('not_contains', 'ne')]:
                with self.subTest(name=name, op=op):
                    result = self.validate(self.request([self.constraint(wrong, op, name)]), name)
                    self.assertEqual(result['filters'], [dict(field=correct, operator=canonical, value=name)])

    def test_moods_remain_soft_and_never_become_accidental_description_keywords(self):
        result = self.validate(self.request([]), 'Make it moving and thoughtful')
        self.assertEqual(result['soft_query'], 'moving and thoughtful')
        self.assertEqual(result['filters'], [])
        with self.assertRaisesRegex(ValueError, 'quoted'):
            self.validate(self.request([self.constraint('description', 'contains', 'moving')]), 'Make it moving')

    def test_numeric_and_boolean_constraints_use_same_typed_validation(self):
        filters = [self.constraint('year', 'gt', '2010'), self.constraint('adult', 'eq', 'false')]
        self.assertEqual(len(self.validate(self.request(filters), 'after 2010; adult false')['filters']), 2)
        with self.assertRaisesRegex(ValueError, 'Numeric'):
            self.validate(self.request([self.constraint('year', 'gt', 'recent')]), 'recent')

    def test_invented_features_and_unsourced_constraints_are_rejected(self):
        for item, question in [(self.constraint('tags', 'ne', 'Imaginary Theme'), 'Imaginary Theme'),
                               (self.constraint('genres', 'eq', 'Fantasy'), 'Suggest something moving')]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.validate(self.request([item]), question)

    def test_broader_mappings_are_visible_as_approximations(self):
        result = self.validate(self.request([self.constraint('tags', 'ne', 'School', 'high school', 'broader')],
                                limitations=['School also covers other education settings.']), 'not set in high school')
        self.assertEqual(result['mappings'][0]['mapping'], 'broader')
        self.assertTrue(result['limitations'])

    def test_assistant_prose_cannot_authorize_inherited_filters(self):
        request = self.request([self.constraint('genres', 'eq', 'Fantasy')], use_previous_context=True)
        with self.assertRaises(ValueError):
            validate_request(request, 'more please', [{'role': 'assistant', 'content': 'Fantasy'}],
                             BASE_FIELDS, self.features)

    def test_mentioned_feature_coverage_uses_the_catalog_not_a_fixed_word_list(self):
        question = 'Fantasy stories excluding military themes, preferably moving'
        mentions = mentioned_features(question, self.features)
        self.assertEqual({name for _, _, name in mentions}, {'Fantasy', 'Military'})
        features = self.features + [('tag', 'new category', 'New Category')]
        self.assertEqual(mentioned_features('exclude New Category', features)[0][2], 'New Category')

    def test_scalar_values_cannot_be_replaced_with_unrelated_concepts(self):
        with self.assertRaisesRegex(ValueError, 'Unknown scalar category'):
            validate_request(self.request([self.constraint('format', 'ne', 'Military')]), 'Military', [],
                             BASE_FIELDS, self.features, {'format': ['TV', 'MOVIE']})

    def test_compound_features_do_not_silently_activate_their_parent_words(self):
        features = self.features + [('tag', 'school club', 'School Club')]
        self.assertEqual([r[2] for r in mentioned_features('without School Club', features)], ['School Club'])
        self.assertEqual({r[2] for r in mentioned_features('School and School Club', features)}, {'School', 'School Club'})
