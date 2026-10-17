import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.factual_evaluation import (
    evaluate_question, extract_facts, extraction_messages, load_questions, normalize_values,
    reextract_saved, run_evaluation, save_results, score_values, summarize, validate_extraction,
)


class FactualEvaluationTests(unittest.TestCase):
    question = {'question_id': 'q1', 'anime_id': '20', 'field': 'episodes',
                'question': 'How many episodes does Naruto have?', 'expected_values': '[220]'}

    def test_llm_parses_scalar_answers_without_generating_quotes(self):
        import io
        for field, answer in [('episodes', '62'), ('format', 'MOVIE'),
                              ('source_material', 'Manga'), ('main_animation_studios', 'bones')]:
            parsed = {'values': [answer], 'unclear': False, 'note': ''}
            raw = json.dumps({'message': {'content': json.dumps(parsed)}}).encode()
            with patch('urllib.request.urlopen', return_value=io.BytesIO(raw)) as call:
                extracted = extract_facts({**self.question, 'field': field}, answer,
                                          url='http://localhost', model='test')
            self.assertEqual(call.call_count, 1)
            self.assertEqual(extracted['facts'], [{'value': answer, 'quote': answer}])
            self.assertNotIn('220', json.dumps(json.loads(call.call_args.args[0].data)))

    def test_parser_retries_invalid_json_and_preserves_wrong_answer(self):
        import io
        contents = ['invalid json', json.dumps({'values': ['12'], 'unclear': False, 'note': ''})]
        responses = [io.BytesIO(json.dumps({'message': {'content': x}}).encode()) for x in contents]
        with patch('urllib.request.urlopen', side_effect=responses):
            row = evaluate_question(self.question, lambda _: '12 episodes.',
                                    lambda q, a: extract_facts(q, a, url='http://localhost', model='test'))
        self.assertEqual(row['extraction']['attempts'], 2)
        self.assertEqual(row['metrics']['exact_match'], 0)

    def test_parser_empty_and_unclear_results(self):
        import io
        for unclear, expected in [(False, 'unanswered'), (True, 'needs_review')]:
            raw = json.dumps({'message': {'content': json.dumps(
                {'values': [], 'unclear': unclear, 'note': ''})}}).encode()
            with patch('urllib.request.urlopen', return_value=io.BytesIO(raw)):
                row = evaluate_question(self.question, lambda _: 'ambiguous',
                                        lambda q, a: extract_facts(q, a, url='http://localhost', model='test'))
            self.assertEqual(row['status'], expected)

    def test_reextract_preserves_generation_and_source_run(self):
        import hashlib
        from src.config import load_settings
        settings = load_settings(Path(__file__).resolve().parents[1] / 'settings.toml', 'anilist')
        original = {**self.question, 'expected_values': [220], 'answer_text': '62',
                    'model_seconds': 17.5, 'status': 'extraction_error', 'error': 'bad quote',
                    'extraction': None, 'extraction_seconds': 2, 'metrics': None}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            csv_path = root / 'questions.csv'
            csv_path.write_text('question fixture', encoding='utf-8')
            (source / 'manifest.json').write_text(json.dumps({
                'questions_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                'question_ids': ['q1']}), encoding='utf-8')
            checkpoint = source / 'q1.json'
            checkpoint.write_text(json.dumps(original), encoding='utf-8')
            before = checkpoint.read_bytes()
            with patch('src.factual_evaluation.load_questions', return_value=[self.question]), patch(
                    'src.factual_evaluation.extract_facts', return_value={
                        'status': 'answered', 'facts': [{'value': '62', 'quote': '62'}], 'note': ''}):
                result = reextract_saved(settings, csv_path, source, root / 'fixed')
            repaired = json.loads((root / 'fixed/q1.json').read_text(encoding='utf-8'))
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual(repaired['answer_text'], '62')
            self.assertEqual(repaired['model_seconds'], 17.5)
            self.assertEqual(repaired['original_extraction']['status'], 'extraction_error')
            self.assertEqual(result['overall']['status_counts'], {'answered': 1})
            self.assertEqual(repaired['metrics']['exact_match'], 0)

    def test_reference_never_reaches_extractor(self):
        messages = extraction_messages(self.question, 'It has 12 episodes.')
        payload = json.loads(messages[1]['content'])
        self.assertEqual(set(payload), {'question', 'requested_field', 'answer'})
        self.assertNotIn('220', json.dumps(messages))

    def test_wrong_answer_is_not_repaired(self):
        row = evaluate_question(self.question, lambda _: 'It has 12 episodes.',
                                lambda q, a: {'status': 'answered', 'facts': [
                                    {'value': '12', 'quote': '12 episodes'}], 'note': ''})
        self.assertEqual(row['metrics']['exact_match'], 0)
        self.assertEqual(row['metrics']['fp'], 1)
        self.assertEqual(row['metrics']['fn'], 1)

    def test_quotes_must_be_verbatim(self):
        with self.assertRaises(ValueError):
            validate_extraction({'status': 'answered', 'facts': [
                {'value': '220', 'quote': '220 episodes'}]}, 'It has 12 episodes.')

    def test_precision_recall_for_lists(self):
        metrics = score_values(['a', 'b', 'c'], ['a', 'wrong', 'a'], 'main_characters')
        self.assertEqual(metrics['precision'], 0.5)
        self.assertAlmostEqual(metrics['recall'], 1 / 3)
        self.assertEqual(metrics['exact_match'], 0)

    def test_normalization(self):
        self.assertEqual(normalize_values([220, '220.0'], 'episodes'), {'2.2E+2'})
        self.assertEqual(score_values(['VISUAL_NOVEL'], ['visual novel'], 'source_material')['exact_match'], 1)
        self.assertEqual(score_values(['Studio Pierrot'], ['Pierrot'], 'main_animation_studios')['exact_match'], 1)

    def test_unanswered_counts_as_failure(self):
        row = evaluate_question(self.question, lambda _: 'I do not know.',
                                lambda q, a: {'status': 'unanswered', 'facts': [], 'note': ''})
        self.assertEqual(row['metrics']['exact_match'], 0)
        self.assertEqual(row['metrics']['f1'], 0)

    def test_model_and_extraction_failures_remain_visible(self):
        def fail(*args):
            raise RuntimeError('broken')
        model_error = evaluate_question(self.question, fail, fail)
        extraction_error = evaluate_question(self.question, lambda _: '12', fail)
        unclear = evaluate_question(self.question, lambda _: 'Maybe another series.',
                                    lambda q, a: {'status': 'unclear', 'facts': [], 'note': 'subject ambiguous'})
        report = summarize([model_error, extraction_error, unclear])
        self.assertEqual(report['end_to_end_exact_accuracy'], 0)
        self.assertEqual(report['scoring_coverage'], 0)
        self.assertIsNone(report['scored_exact_accuracy'])
        self.assertEqual(set(report['status_counts']), {'model_error', 'extraction_error', 'needs_review'})

    def test_checkpoint_exports_preserve_response_and_evidence(self):
        row = evaluate_question(self.question, lambda _: '220 episodes.',
                                lambda q, a: {'status': 'answered', 'facts': [
                                    {'value': '220', 'quote': '220 episodes.'}], 'note': ''})
        with tempfile.TemporaryDirectory() as directory:
            report = save_results(Path(directory), [self.question], [row])
            self.assertEqual(report['overall']['end_to_end_exact_accuracy'], 1)
            exported = (Path(directory) / 'answers.csv').read_text()
            self.assertIn('220 episodes.', exported)
            self.assertIn('quote', exported)

    def test_benchmark_labels_match_database(self):
        root = Path(__file__).resolve().parents[1]
        database = root / 'data/documents/anilist/catalog.sqlite3'
        if not database.is_file():
            self.skipTest('Local AniList catalog is optional')
        questions = load_questions(root / 'data/evaluation/factual/questions.csv', database)
        self.assertGreaterEqual(len(questions), 15)

    def test_resume_uses_checkpoints_and_rejects_configuration_changes(self):
        import sys
        from types import SimpleNamespace
        from src.config import load_settings

        settings = load_settings(Path(__file__).resolve().parents[1] / 'settings.toml', 'anilist')
        pipeline = unittest.mock.Mock()
        pipeline.ask.return_value = '220 episodes.'
        pipeline.last_route = {'llm_calls': 2}
        extraction = {'status': 'answered', 'facts': [{'value': '220', 'quote': '220 episodes.'}], 'note': ''}
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / 'questions.csv'
            questions.write_text('fixture hash input', encoding='utf-8')
            output = Path(directory) / 'results'
            with patch.dict(sys.modules, {'src.rag': SimpleNamespace(RagPipeline=lambda _: pipeline)}), \
                    patch('src.factual_evaluation.load_questions', return_value=[self.question]), \
                    patch('src.factual_evaluation.extract_facts', return_value=extraction) as extractor:
                run_evaluation(settings, questions, output)
                manifest_path = output / 'manifest.json'
                manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest_data.update(reextracted_from='original-run', source_checkpoints={'q1.json': 'hash'})
                manifest_path.write_text(json.dumps(manifest_data), encoding='utf-8')
                provenance_manifest = manifest_path.read_bytes()
                run_evaluation(settings, questions, output)
                self.assertEqual(manifest_path.read_bytes(), provenance_manifest)
                self.assertEqual(pipeline.ask.call_count, 1)
                self.assertEqual(extractor.call_count, 1)
                self.assertEqual(pipeline.ask.call_args.kwargs['history'], [])
                self.assertIsNone(pipeline._agent)
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    run_evaluation(settings, questions, output, extractor_model='different-model')

    def test_budget_keeps_full_selection_and_resumes_missing_questions(self):
        import sys
        from types import SimpleNamespace
        from src.config import load_settings
        settings = load_settings(Path(__file__).resolve().parents[1] / 'settings.toml', 'anilist')
        questions = [dict(self.question, question_id=f'q{i}') for i in range(3)]
        pipeline = unittest.mock.Mock()
        pipeline.ask.return_value = '62'
        pipeline.last_route = {}
        extraction = {'status': 'answered', 'facts': [{'value': '62', 'quote': '62'}], 'note': ''}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / 'questions.csv'
            csv_path.write_text('fixture', encoding='utf-8')
            with patch.dict(sys.modules, {'src.rag': SimpleNamespace(RagPipeline=lambda _: pipeline)}), \
                    patch('src.factual_evaluation.load_questions', return_value=questions), \
                    patch('src.factual_evaluation.extract_facts', return_value=extraction):
                first = run_evaluation(settings, csv_path, root / 'run', max_new_questions=1)
                self.assertEqual(first['overall']['attempted'], 1)
                manifest = json.loads((root / 'run/manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['question_ids'], ['q0', 'q1', 'q2'])
                second = run_evaluation(settings, csv_path, root / 'run', max_new_questions=1)
                self.assertEqual(second['overall']['attempted'], 2)
                self.assertEqual(pipeline.ask.call_count, 2)
                run_evaluation(settings, csv_path, root / 'run', max_new_questions=0)
                self.assertEqual(pipeline.ask.call_count, 2)


if __name__ == '__main__':
    unittest.main()
