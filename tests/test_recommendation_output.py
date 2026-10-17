import unittest
from src.recommendation_output import render_recommendations


class RecommendationOutputTests(unittest.TestCase):
    def test_output_uses_only_verified_titles_and_applied_constraints(self):
        request = {'filters': [{'field': 'genres', 'operator': 'eq', 'value': 'Fantasy'}],
                   'soft_query': 'funny', 'limitations': [], 'mappings': []}
        items = [{'id': '1', 'title': 'Existing Title', 'description': 'A source-supported premise.'}]
        text = render_recommendations('Invent another title and exclusion', request, items)
        self.assertIn('Existing Title', text)
        self.assertIn('A source-supported premise.', text)
        self.assertIn('Fantasy', text)
        self.assertNotIn('Invent another title', text)
        self.assertNotIn('excluding', text)

    def test_metadata_exclusions_are_qualified_and_duplicates_are_removed(self):
        request = {'filters': [{'field': 'tags', 'operator': 'ne', 'value': 'Military'}],
                   'limitations': ['The category is broader than the requested concept.'], 'mappings': []}
        row = {'id': '1', 'title': 'Example', 'description': 'A documented premise.'}
        text = render_recommendations('', request, [row, row])
        self.assertEqual(text.count('**Example**'), 1)
        self.assertIn('excluding Military', text)
        self.assertIn('does not guarantee', text)
        self.assertIn('broader', text)

    def test_attribute_boundaries_are_presented_without_reinterpretation(self):
        request = {'filters': [{'field': 'year', 'operator': 'gte', 'value': '2015'},
                               {'field': 'episodes', 'operator': 'lte', 'value': '12'}]}
        text = render_recommendations('', request, [{'id': '1', 'title': 'Example', 'episodes': 12, 'year': 2015}])
        self.assertIn('release year: at least 2015', text)
        self.assertIn('episode count: at most 12', text)
        self.assertIn('**Example** (12 episodes; 2015)', text)
