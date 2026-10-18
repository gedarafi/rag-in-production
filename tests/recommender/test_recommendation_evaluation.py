"""Hand-computed metric cases and failure-path checks; no model/network needed."""
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.recommender.recommendation_evaluation import (
    Catalog, constraint_score, evaluate_question, extract_titles, extraction_messages,
    load_dataset, rescore_saved, run_evaluation, score_recommendations, summarize, validate_constraint,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/evaluation/recommendations'


class RecommendationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        records = [
            {'id': '1', 'title': 'First Show', 'aliases': ['Ichi', 'Shared'], 'tag_ranks': {'Calm': 100},
             'genres': ['Fantasy'], 'year': 2020, 'start_date': '2020-01-02', 'studios': ['Studio A'],
             'format': 'TV', 'source_material': 'MANGA', 'episodes': 12, 'duration': 24},
            {'id': '2', 'title': 'Second Show', 'aliases': ['Ni', 'Shared'], 'tag_ranks': {'Calm': 50},
             'genres': ['Fantasy'], 'year': 1999, 'studios': ['Studio B']},
            {'id': '3', 'title': 'Third Show', 'aliases': [], 'tag_ranks': {'Sad': 90}, 'genres': ['Comedy']},
        ]
        path = Path(self.temporary.name) / 'catalog.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        self.catalog = Catalog(path)
        self.q = {'question_id': 'q1', 'question': 'Recommend three calming anime.', 'k': 3,
                  'category': 'tags', 'wording': 'indirect',
                  'constraints': [{'field': 'tags', 'op': 'any', 'value': ['Calm']}]}

    def test_title_aliases_collisions_and_no_fuzzy_season_guessing(self):
        self.assertEqual(self.catalog.resolve('**FIRST-SHOW!**')[:2], ('1', 'resolved'))
        self.assertEqual(self.catalog.resolve('Ichi')[:2], ('1', 'resolved'))
        self.assertEqual(self.catalog.resolve('Shared'), (None, 'ambiguous', ['1', '2']))
        self.assertEqual(self.catalog.resolve('First Show Season 2')[1], 'unknown')

    def test_hand_computed_scores_missing_slots_and_ideal_ranking(self):
        metrics, _ = score_recommendations(self.q, ['Second Show', 'First Show'], self.catalog)
        self.assertAlmostEqual(metrics['precision_at_k'], 2 / 3)
        self.assertAlmostEqual(metrics['graded_fit_at_k'], .5)
        self.assertAlmostEqual(metrics['ndcg_at_k'], (.5 + 1 / math.log2(3)) / (1 + .5 / math.log2(3)))
        self.assertEqual(metrics['constraint_coverage'], 1)
        self.assertEqual(metrics['count_compliance'], 0)
        ideal, _ = score_recommendations(self.q, ['First Show', 'Second Show', 'Third Show'], self.catalog)
        self.assertAlmostEqual(ideal['ndcg_at_k'], 1.)

    def test_duplicate_alias_unknown_and_extras_do_not_inflate_scores(self):
        metrics, details = score_recommendations(self.q, ['First Show', 'Ichi', 'Invented', 'Second Show'], self.catalog)
        self.assertAlmostEqual(metrics['precision_at_k'], 1 / 3)
        self.assertAlmostEqual(metrics['graded_fit_at_k'], 1 / 3)
        self.assertEqual(metrics['extra_count'], 1)
        self.assertEqual(metrics['duplicate_rate'], .25)
        self.assertEqual(metrics['resolution_rate'], .75)
        self.assertTrue(details[1]['duplicate'])
        self.assertEqual(details[1]['raw_fit'], 1)
        self.assertEqual(details[1]['gain'], 0)
        self.assertEqual(details[2]['strict_match'], 0)

    def test_multiple_constraints_coverage_is_distinct_from_joint_compliance(self):
        q = {**self.q, 'k': 2, 'constraints': [*self.q['constraints'], {'field': 'genres', 'op': 'any', 'value': ['Comedy']}]}
        metrics, _ = score_recommendations(q, ['First Show', 'Third Show'], self.catalog)
        self.assertEqual(metrics['precision_at_k'], 0)
        self.assertEqual(metrics['constraint_coverage'], 1)
        self.assertEqual(metrics['graded_fit_at_k'], .5)

    def test_tag_alternatives_exclusion_threshold_and_missing_scalar(self):
        r = self.catalog.records['2']
        q = {'constraints': [{'field': 'tags', 'op': 'none', 'value': ['Calm']}]}
        self.assertEqual(self.catalog.fits(q, r), ([.5], 0))
        self.assertEqual(constraint_score({'field': 'tags', 'op': 'any', 'value': ['Calm', 'Sad']}, r), .5)
        self.assertEqual(constraint_score({'field': 'tags', 'op': 'all', 'value': ['Calm', 'Sad']}, r), 0)
        self.assertEqual(constraint_score({'field': 'episodes', 'op': 'lte', 'value': 12}, r), 0)
        self.assertEqual(constraint_score({'field': 'start_date', 'op': 'gte', 'value': '1990-01-01'}, r), 0)

    def test_studio_source_date_and_runtime_constraints(self):
        q = {'constraints': [
            {'field': 'studios', 'op': 'any', 'value': ['Studio A']},
            {'field': 'source_material', 'op': 'any', 'value': ['manga']},
            {'field': 'start_date', 'op': 'gte', 'value': '2020-01-02'},
            {'field': 'year', 'op': 'lte', 'value': 2020},
            {'field': 'format', 'op': 'any', 'value': ['TV']},
            {'field': 'episodes', 'op': 'lte', 'value': 12},
            {'field': 'duration', 'op': 'eq', 'value': 24}]}
        self.assertEqual(self.catalog.fits(q, self.catalog.records['1']), ([1.] * 7, 1))
        for invalid in [dict(field='year', op='any', value=[2020]), dict(field='studios', op='eq', value='X'),
                        dict(field='start_date', op='gte', value='2020-02-31')]:
            with self.assertRaises(ValueError):
                validate_constraint(invalid)

    def test_reference_labels_never_reach_extractor(self):
        q = {**self.q, 'constraints': [{'secret': 'EXPECTED_LABEL_SECRET'}], 'reference_examples': ['GOLD_SECRET']}
        payload = extraction_messages(q, 'First Show')
        self.assertEqual(set(json.loads(payload[1]['content'])), {'question', 'answer'})
        self.assertNotIn('SECRET', json.dumps(payload))

    def test_mystery_alternative_receives_full_credit_without_detective_tag(self):
        q = {**self.q, 'k': 1, 'constraints': [{'field': 'tags', 'op': 'any', 'value': ['Detective'],
             'alternatives': [{'reason': 'Investigative mystery is a valid interpretation.',
                               'constraints': [{'field': 'genres', 'op': 'any', 'value': ['Mystery']}]}]}]}
        self.catalog.records['1']['genres'] = ['Mystery']
        self.catalog.records['1']['tag_ranks'] = {'Unrelated': 100}
        metrics, details = score_recommendations(q, ['First Show'], self.catalog)
        self.assertEqual(metrics['precision_at_k'], 1)
        self.assertEqual(metrics['graded_fit_at_k'], 1)
        self.assertEqual(metrics['literal_precision_at_k'], 0)
        self.assertEqual(metrics['semantic_credit_at_k'], 1)
        self.assertEqual(details[0]['constraint_evidence'][0]['credited_path'], 'alternative_1')
        self.assertEqual(metrics['ndcg_at_k'], 1)

    def test_alternatives_are_directional_and_do_not_waive_other_requirements(self):
        c = {'field': 'tags', 'op': 'any', 'value': ['Cyberpunk'], 'alternatives': [
            {'reason': 'Harsh technological future', 'constraints': [
                {'field': 'genres', 'op': 'any', 'value': ['Sci-Fi']},
                {'field': 'tags', 'op': 'any', 'value': ['Dystopian']},
                {'field': 'tags', 'op': 'any', 'value': ['Artificial Intelligence']}]}]}
        partial = {'genres': ['Sci-Fi'], 'tag_ranks': {'Dystopian': 100}}
        self.assertEqual(constraint_score(c, partial), 0)
        complete = {**partial, 'tag_ranks': {'Dystopian': 100, 'Artificial Intelligence': 90}}
        self.assertEqual(constraint_score(c, complete), .9)
        self.assertEqual(self.catalog.fits({'constraints': [c]}, complete)[1], 1)
        self.assertEqual(self.catalog.fits({'constraints': [c, {'field': 'year', 'op': 'gte', 'value': 2020}]}, complete)[1], 0)
        literal = {'field': 'tags', 'op': 'any', 'value': ['Detective']}
        self.assertEqual(constraint_score(literal, {'genres': ['Mystery']}), 0)

    def test_alternative_conjunction_preserves_exclusion_boundary(self):
        c = {'field': 'tags', 'op': 'any', 'value': ['Iyashikei'], 'alternatives': [
            {'reason': 'Low stakes everyday life', 'constraints': [
                {'field': 'genres', 'op': 'any', 'value': ['Slice of Life']},
                {'field': 'tags', 'op': 'none', 'value': ['Tragedy']}]}]}
        self.assertEqual(self.catalog.fits({'constraints': [c]}, {'genres': ['Slice of Life'], 'tag_ranks': {'Tragedy': 50}})[1], 0)
        self.assertEqual(self.catalog.fits({'constraints': [c]}, {'genres': ['Slice of Life'], 'tag_ranks': {'Tragedy': 49}})[1], 1)
        with self.assertRaises(ValueError):
            validate_constraint({**c, 'op': 'none'})

    def test_llm_extraction_retries_invalid_or_invented_titles(self):
        for first in ['bad JSON', json.dumps({'titles': ['Made Up'], 'unclear': False, 'note': ''})]:
            values = [first, json.dumps({'titles': ['Ichi', 'Ni', 'Ichi'], 'unclear': False, 'note': ''})]
            responses = [io.BytesIO(json.dumps({'message': {'content': v}}).encode()) for v in values]
            with patch('urllib.request.urlopen', side_effect=responses) as call:
                result = extract_titles(self.q, '1. Ichi\n2. Ni\n3. Ichi', url='http://localhost', model='test')
            self.assertEqual(call.call_count, 2)
            self.assertEqual(result['attempts'], 2)
            self.assertEqual(result['titles'], ['Ichi', 'Ni', 'Ichi'])

    def test_errors_and_unclear_extraction_remain_in_denominator(self):
        def fail(*args):
            raise RuntimeError('broken')
        rows = [evaluate_question(self.q, fail, fail, self.catalog),
                evaluate_question(self.q, lambda _: 'First Show', fail, self.catalog),
                evaluate_question(self.q, lambda _: 'First Show', lambda *a: {'titles': ['First Show'], 'unclear': True, 'note': 'ambiguous'}, self.catalog),
                evaluate_question(self.q, lambda _: 'No suggestions.', lambda *a: {'titles': [], 'unclear': False, 'note': ''}, self.catalog)]
        report = summarize(rows, 8)
        self.assertEqual(report['end_to_end_precision_at_k'], 0)
        self.assertEqual(report['scoring_coverage'], .25)
        self.assertEqual(report['pending'], 4)
        self.assertEqual(report['scored_macro']['precision_at_k'], 0)
        self.assertEqual({r['status'] for r in rows}, {'model_error', 'extraction_error', 'needs_review', 'unanswered'})
        self.assertIsNone(summarize([], 8)['end_to_end_precision_at_k'])


class AuthoredRecommendationInputTests(unittest.TestCase):
    def test_offline_inspection_loads_without_bulk_references_but_live_scoring_requires_them(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            manifest = json.loads((DATA / 'dataset_manifest.json').read_text(encoding='utf-8'))
            for name in ['dataset_manifest.json', *manifest['files_sha256']]:
                if name != 'catalog_reference.jsonl':
                    shutil.copy2(DATA / name, destination / name)
            questions, catalog, _ = load_dataset(destination, require_catalog=False)
            self.assertEqual(len(questions), 72)
            self.assertIsNone(catalog)
            with self.assertRaisesRegex(FileNotFoundError, 'build_recommendation_benchmark'):
                load_dataset(destination)

    def test_small_versioned_inputs_are_intact_without_a_local_catalog(self):
        import hashlib
        manifest = json.loads((DATA / 'dataset_manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files_sha256'].items():
            if name == 'catalog_reference.jsonl':
                continue
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), expected, name)
        questions = [json.loads(line) for line in (DATA / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(questions), manifest['question_count'])
        self.assertEqual(len({q['question_id'] for q in questions}), len(questions))
        for q in questions:
            for c in q['constraints']:
                validate_constraint(c)


@unittest.skipUnless((DATA / 'catalog_reference.jsonl').is_file(),
                     'Build local references with scripts/build_recommendation_benchmark.py after AniList setup')
class DeliveredRecommendationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions, cls.catalog, cls.manifest = load_dataset(DATA)

    def test_authored_dataset_coverage_and_known_satisfiable_constraints(self):
        self.assertEqual(len(self.questions), 72)
        self.assertEqual(len({q['question'].casefold() for q in self.questions}), 72)
        self.assertEqual(sum(q['wording'] == 'indirect' for q in self.questions), 48)
        fields = {c['field'] for q in self.questions for c in q['constraints']}
        self.assertEqual(fields, {'tags', 'genres', 'year', 'start_date', 'studios', 'format', 'source_material', 'episodes', 'duration'})
        for q in self.questions:
            self.assertGreaterEqual(q['strict_catalog_matches'], q['k'])
            self.assertTrue(q['label_rationale'])
            self.assertEqual(q['question_origin'], 'authored_naturalistic')
            for example in q['reference_examples']:
                self.assertEqual(self.catalog.fits(q, self.catalog.records[example['id']])[1], 1)

    def test_zero_budget_does_not_load_rag_and_config_changes_reject_resume(self):
        from src.recommender.config import load_settings
        settings = load_settings(ROOT / 'settings.toml', 'anilist')
        with tempfile.TemporaryDirectory() as folder:
            report = run_evaluation(settings, DATA, folder, max_new_questions=0)
            self.assertEqual(report['overall']['planned'], 11)
            self.assertEqual(report['overall']['attempted'], 0)
            self.assertIsNone(report['overall']['scored_macro']['precision_at_k'])
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                run_evaluation(settings, DATA, folder, extractor_model='different', max_new_questions=0)

    def test_policy_accepts_real_mystery_only_catalog_candidates(self):
        lookup = {q['question_id']: q for q in self.questions}
        indirect = lookup['detectives_02']
        literal = lookup['detectives_01']
        candidates = [r for r in self.catalog.records.values()
                      if 'Mystery' in r['genres'] and r['tag_ranks'].get('Detective', 0) < 50]
        self.assertTrue(candidates)
        for r in candidates[:20]:
            self.assertEqual(self.catalog.fits(indirect, r)[1], 1)
            self.assertEqual(self.catalog.fits(literal, r)[1], 0)
        self.assertGreater(self.manifest['catalog_tag_count'], 300)

    def test_specific_comedy_theme_can_satisfy_a_general_named_genre_request(self):
        q = next(q for q in self.questions if q['question_id'] == 'comedy_01')
        record = {'genres': [], 'tag_ranks': {'Surreal Comedy': 100, 'Space': 100}}
        self.assertEqual(self.catalog.fits(q, record)[1], 1)
        self.assertEqual(constraint_score(q['constraints'][0], record), 1)
        self.assertEqual(self.catalog.fits(q, {'genres': ['Mystery'], 'tag_ranks': {'Space': 100}})[1], 0)

    def test_rescoring_preserves_original_answer_and_llm_extraction(self):
        q = next(q for q in self.questions if q['question_id'] == 'detectives_02')
        record = next(r for r in self.catalog.records.values() if 'Mystery' in r['genres'] and not r['tag_ranks'].get('Detective'))
        row = {**q, 'answer_text': record['title'], 'status': 'scored', 'error': '',
               'metrics': {'precision_at_k': 0}, 'recommendations': [], 'model_seconds': 12,
               'extraction_seconds': 3, 'extraction': {'titles': [record['title']], 'unclear': False, 'note': '', 'method': 'llm'}}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            source.mkdir()
            (source / 'manifest.json').write_text(json.dumps({'question_ids': [q['question_id']], 'evaluator_version': '1'}), encoding='utf-8')
            checkpoint = source / (q['question_id'] + '.json')
            checkpoint.write_text(json.dumps(row), encoding='utf-8')
            before = checkpoint.read_bytes()
            output = Path(folder) / 'revised'
            report = rescore_saved(DATA, source, output)
            revised = json.loads((output / checkpoint.name).read_text(encoding='utf-8'))
            self.assertEqual(checkpoint.read_bytes(), before)
            for name in ('answer_text', 'extraction', 'model_seconds', 'extraction_seconds'):
                self.assertEqual(row[name], revised[name])
            self.assertGreater(report['overall']['scored_macro']['semantic_credit_at_k'], 0)


if __name__ == '__main__':
    unittest.main()
