import json
import tempfile
import unittest
from pathlib import Path

from src.recommender.anime_tools import AnimeTools, structured
from src.recommender.query_router import QueryPlan, fast_answer


class CharacterIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        records = []
        for aid, title, popularity, cid, name in [
            ('1', 'Re:Zero', 100, 17, 'Subaru Natsuki'),
            ('2', 'Maid-Sama!', 900, 18, 'Subaru'),
            ('3', 'Zero', 10000, 19, 'Subaru')]:
            row = {'id': aid, 'title_english': title, 'popularity': popularity,
                   'characters': [{'role': 'MAIN' if cid == 17 else 'SUPPORTING', 'node': {
                       'id': cid, 'name': {'full': name}, 'description': title + ' character biography.'}}]}
            records.append(structured(row, f'anime-{aid}.md'))
        (folder / 'catalog.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        self.tools = AnimeTools.load(folder)

    def test_reversed_full_name_beats_more_popular_partial_matches(self):
        for name in ('Natsuki Subaru', 'Subaru Natsuki', 'Who is Natsuki Subaru?'):
            found = self.tools.character(name)
            self.assertEqual(found[0]['id'], 17)
            self.assertEqual(found[1]['title'], 'Re:Zero')

    def test_unknown_full_name_does_not_degrade_to_one_token(self):
        self.assertIsNone(self.tools.character('Invented Subaru'))

    def test_franchise_scope_excludes_short_incidental_title_aliases(self):
        self.assertEqual(self.tools.character('Subaru', anime_scope='Re:Zero')[0]['id'], 17)
        self.assertIsNone(self.tools.character('Natsuki Subaru', anime_scope='Maid-Sama!'))
        self.assertIsNone(self.tools.character('Natsuki Subaru', anime_scope='Unknown franchise'))

    def test_character_tool_executes_scope_without_a_global_fallback(self):
        plan = QueryPlan('character', 'Natsuki Subaru', entity_type='character', anime_scope='Re:Zero')
        answer = fast_answer(self.tools, 'I mean from Re:Zero', plan=plan)
        self.assertIn('Subaru Natsuki', answer)
        self.assertIn('[anime-1.md]', answer)
        self.assertNotIn('Maid-Sama', answer)


if __name__ == '__main__':
    unittest.main()
