"""Check the actual failed request and a year-only control, without modifying old runs."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommender.config import load_settings
from src.recommender.evaluation_runtime import traced_asker
from src.recommender.rag import RagPipeline
from src.recommender.recommendation_evaluation import VERSION, evaluate_question, extract_titles, load_dataset, save_results

DATA = ROOT / 'data/evaluation/recommendations'
OUTPUT = DATA / 'results/date_fix_validation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='New directory for live verification results')
    output = parser.parse_args().output
    questions, catalog, dataset_manifest = load_dataset(DATA)
    settings = load_settings(ROOT / 'settings.toml', 'anilist')
    selected = [next(q for q in questions if q['question_id'] == identifier)
                for identifier in ('dates_01', 'recent_01')]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps({
        'settings': json.loads(json.dumps(asdict(settings), default=str)),
        'dataset_files': dataset_manifest['files_sha256'], 'question_ids': [q['question_id'] for q in selected],
        'extractor_model': settings.llm, 'evaluator_version': VERSION, 'mode': 'date_fix_validation',
    }, indent=2), encoding='utf-8')
    pipeline = RagPipeline(settings)
    rows = []
    for q in selected:
        pipeline._agent = None
        events = []
        print(q['question_id'], q['question'], flush=True)
        row = evaluate_question(q, traced_asker(pipeline, events),
                                lambda q, a: extract_titles(q, a, url=settings.ollama_url, model=settings.llm), catalog)
        row.update(debug_events=events, rag_usage=pipeline.last_route)
        (output / (q['question_id'] + '.json')).write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        rows.append(row)
        save_results(output, selected, rows)
        print(row['status'], row['error'], row['metrics'], flush=True)
    for row in rows:
        assert row['status'] == 'scored', (row['question_id'], row['error'])
        assert row['metrics']['precision_at_k'] == 1, (row['question_id'], row['recommendations'])
        assert not any(e.get('stage') == 'tool' and e.get('status') == 'error' for e in row['debug_events']), row['question_id']
    print('Failed full-date request and existing year-only request both passed live evaluation.', flush=True)


if __name__ == '__main__':
    main()
