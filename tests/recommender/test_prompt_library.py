import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jinja2 import UndefinedError

from src.recommender.config import load_settings
from src.recommender.entity_agent import TOOLS, EntityAgent
from src.recommender.prompt_library import PACKAGED, PromptLibrary, library, prompts_for


class PromptLibraryTests(unittest.TestCase):
    def test_every_packaged_template_parses(self):
        environment = library().environment
        names = sorted(path.name for path in PACKAGED.glob('*.j2'))
        self.assertGreater(len(names), 20)
        for name in names:
            with self.subTest(template=name):
                environment.parse((PACKAGED / name).read_text(encoding='utf-8'))

    def test_rendered_text_is_the_file_without_its_final_newline(self):
        text = (PACKAGED / 'agent_reply_system.j2').read_text(encoding='utf-8')
        self.assertTrue(text.endswith('\n'))
        self.assertEqual(library().render('agent_reply_system'), text[:-1])

    def test_a_missing_value_fails_instead_of_sending_an_empty_gap(self):
        with self.assertRaises(UndefinedError):
            library().render('concept_role', question='Find games without combat')

    def test_conditional_parts_follow_their_flags(self):
        grounded = library().render('agent_step_system', grounded=True)
        self.assertTrue(grounded.startswith('Source evidence HAS ALREADY BEEN LOADED'))
        self.assertTrue(library().render('agent_step_system', grounded=False).startswith('No factual evidence'))

    def test_override_folder_replaces_only_its_own_files(self):
        with tempfile.TemporaryDirectory() as folder:
            overrides = Path(folder)
            (overrides / 'agent_reply_system.j2').write_text('Answer like a pirate.\n', encoding='utf-8')
            (overrides / 'tools.json').write_text(json.dumps({name: 'Custom ' + name for name in TOOLS}),
                                                  encoding='utf-8')
            prompts = PromptLibrary(overrides)
            self.assertEqual(prompts.render('agent_reply_system'), 'Answer like a pirate.')
            self.assertEqual(prompts.render('plan_tools_system'), library().render('plan_tools_system'))
            self.assertEqual(prompts.data('tools.json')['popular'], 'Custom popular')

    def test_settings_choose_the_override_folder(self):
        settings = load_settings()
        self.assertIs(prompts_for(settings), library())
        with tempfile.TemporaryDirectory() as folder:
            chosen = prompts_for(replace(settings, prompt_overrides=Path(folder)))
            self.assertEqual(chosen.folders, [Path(folder), PACKAGED])

    def test_tool_descriptions_must_cover_exactly_the_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'tools.json').write_text(json.dumps({'popular': 'Rank anime.'}), encoding='utf-8')
            agent = EntityAgent.__new__(EntityAgent)  # only its settings matter here
            agent.settings = replace(load_settings(), prompt_overrides=Path(folder))
            with self.assertRaisesRegex(ValueError, 'tools.json'):
                agent.tool_guides
        self.assertEqual(list(library().data('tools.json')), list(TOOLS))


if __name__ == '__main__':
    unittest.main()
