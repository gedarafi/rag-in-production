"""Update hard-facts notebook sources while retaining saved execution outputs."""
import json
from pathlib import Path
from notebook_utils import write_notebook

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = dict(cell_type=kind, id=f'hard-facts-{len(cells):02d}', metadata={},
                source=source.strip().splitlines(keepends=True))
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell('markdown', '''
# Hard facts: 180 natural-language questions

Validate the frozen CSV, run the actual anime RAG, and evaluate requested facts.
**180 authored English prompts across 76 anime; no repeated question or anime/field pair.**
These are naturalistic synthetic questions, not user logs or an external question dataset.
Popular titles are overrepresented; this is not a representative sample of user traffic.

Seven fields: release **year**, episode count, minutes per episode, format, source medium,
main animation studios, and main characters. Full calendar dates, unknown-answer
abstention, conversational memory, and recommendations are outside this benchmark.
See `data/evaluation/hard_facts/README.md` for provenance and definitions.
''')
cell('code', '''
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import subprocess
import sys

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src/factual_evaluation.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Open the notebook from the repository root or notebook directory.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string(index=False) if hasattr(value, 'to_string') else value)
from src.config import load_settings
from src.factual_evaluation import load_questions, read_csv, score_values

DATA = ROOT / 'data/evaluation/hard_facts'
QUESTIONS_CSV = DATA / 'questions.csv'
SETTINGS = load_settings(ROOT / 'settings.toml', 'anilist')
DATABASE = SETTINGS.documents / 'catalog.sqlite3'
PROJECT_PYTHON = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not PROJECT_PYTHON.is_file():
    PROJECT_PYTHON = Path(sys.executable)
manifest = json.loads((DATA / 'dataset_manifest.json').read_text(encoding='utf-8'))
questions = load_questions(QUESTIONS_CSV, DATABASE if DATABASE.is_file() else None)
frame = pd.DataFrame(questions)
print(f'{len(questions)} questions | {frame.anime_id.nunique()} anime | {frame.field.nunique()} fields')
print('Notebook kernel:', sys.executable)
print('Live evaluation interpreter:', PROJECT_PYTHON)
''')
cell('markdown', '''
## Dataset integrity and coverage

Labels are independently queried from the imported local snapshot, not live websites.
Studios require both `main=true` and `animation=true`; characters require role `MAIN`.
Season-specific entries are scored separately. Rebuilding labels is an intentional
versioning operation, never part of evaluation. References below are for inspection;
they are not passed to the RAG or the claim extractor.
''')
cell('code', '''
assert len(questions) == manifest['question_count'] == 180
assert frame.question_id.is_unique and frame.question.str.casefold().is_unique
assert not frame.duplicated(['anime_id', 'field']).any()
assert frame.anime_id.nunique() == manifest['unique_anime']
assert hashlib.sha256(QUESTIONS_CSV.read_bytes()).hexdigest() == manifest['questions_sha256']
evidence = [{'question_id': q['question_id'], 'anime_id': q['anime_id'],
             'field': q['field'], 'values': json.loads(q['expected_values'])} for q in questions]
assert hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == manifest['labels_sha256']
assert Counter(frame.field) == manifest['by_field']
assert frame.question_origin.eq('authored_naturalistic').all()
display(frame.groupby('field').agg(questions=('question_id', 'size'), anime=('anime_id', 'nunique')).reset_index())
display(frame.groupby('field', sort=False).head(2)[['question_id', 'field', 'question', 'expected_values']])
if DATABASE.is_file():
    print('All frozen labels match:', DATABASE)
else:
    print('Frozen input hashes verified. Database comparison is available after AniList setup; offline inspection needs no bulk data.')
''')
cell('markdown', '''
## Dataset visualization

Coverage is shown for the complete CSV. Charts are rendered with matplotlib in the
project environment and saved as PNG and SVG, so the notebook kernel does not need
matplotlib. Install the chart dependencies once with `uv sync --extra evaluation`.
''')
cell('code', '''
def show_evaluation_charts(results=None):
    chart_output = (Path(results) / 'charts') if results is not None else DATA / 'charts'
    command = [str(PROJECT_PYTHON), '-X', 'utf8', '-m', 'src.hard_facts_visualizations',
               '--questions', str(QUESTIONS_CSV), '--output', str(chart_output)]
    if results is not None:
        command.extend(['--results', str(results)])
    chart_environment = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONIOENCODING': 'utf-8'}
    completed = subprocess.run(command, cwd=ROOT, env=chart_environment, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
    if completed.returncode:
        hint = 'Run uv sync --extra evaluation. ' if 'ModuleNotFoundError' in completed.stderr else ''
        raise RuntimeError('Chart rendering failed. ' + hint + 'Details: ' + completed.stderr)
    artifacts = json.loads((chart_output / 'charts.json').read_text(encoding='utf-8'))
    for artifact in artifacts:
        # The dataset-only view already displays this chart.
        if results is not None and Path(artifact['png']).stem == 'dataset_coverage':
            continue
        try:
            from IPython.display import Image
        except ImportError:
            print('Chart:', artifact['png'])
        else:
            display(Image(filename=artifact['png']))
        print(artifact['caption'])
    print('PNG and SVG exports:', chart_output)
    return artifacts

dataset_charts = show_evaluation_charts()
''')
cell('markdown', '''
## Experiment configuration

Set `RUN_LIVE = True` to call the model. Choose `MODE = 'full'` and a new `RUN_NAME`
to evaluate all 180 prompts. Smoke mode selects one prompt per field (seven total).
The default validates data and displays any saved smoke run without calling the model.
The notebook kernel needs pandas and Python 3.11 or newer. Live evaluation always runs
in the project's `.venv` interpreter, even when Jupyter uses a different Python.
This avoids importing incompatible global packages such as the obsolete `docx` package.
Ollama and prepared AniList indexes must already be available.
No setup rebuild or downloads are performed. A full run may take substantial time.

Each finished question is checkpointed. An identical run resumes, including failures;
use a fresh run name to retry or change configuration. `EXTRACTOR_MODEL` can name a
separately installed stronger model to reduce correlated generation/extraction errors.
''')
cell('code', '''
RUN_LIVE = False
MODE = 'smoke'                # 'full' runs all 180 questions
RUN_NAME = 'smoke_validation' # use e.g. 'full_001' for a full experiment
EXTRACTOR_MODEL = None        # defaults to SETTINGS.llm
MAX_NEW_QUESTIONS = None      # optional per-execution budget; None continues all missing prompts

if MODE not in {'smoke', 'full'}:
    raise ValueError("MODE must be 'smoke' or 'full'")
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {'.', '..'}:
    raise ValueError('RUN_NAME must be a single directory name')
first_per_field = {}
for question in questions:
    first_per_field.setdefault(question['field'], question)
selected = questions if MODE == 'full' else list(first_per_field.values())
RUN_QUESTIONS = QUESTIONS_CSV
if MODE == 'smoke':
    RUN_QUESTIONS = DATA / 'smoke_questions.csv'
    with RUN_QUESTIONS.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(questions[0]))
        writer.writeheader()
        writer.writerows(selected)
OUTPUT = DATA / 'results' / RUN_NAME
print(f'Mode: {MODE}; selected: {len(selected)}; live inference: {RUN_LIVE}')
print('Generation:', SETTINGS.llm, '| Extractor:', EXTRACTOR_MODEL or SETTINGS.llm)
print('Output:', OUTPUT)
''')
cell('markdown', '''
## Run inference and report accuracy

Only question text reaches `RagPipeline.ask`, with empty history and fresh agent state.
The extractor receives the question, requested field, and final answer; it never sees
reference values or catalog records. It parses asserted values through a dedicated LLM request. Code attaches the full original answer as evidence.
Deterministic set comparison penalizes missing and extra values; no fuzzy similarity.

**Primary metric: exact correct answers / all attempted questions.** Model errors,
extraction errors, and unclear extractions count as unsuccessful primary-metric attempts.
Macro value precision/recall/F1 apply to scored questions; unanswered answers score zero.
Always inspect scoring coverage. Only the requested field is evaluated, not every
additional prose claim. Startup is included in the first RAG latency; extraction is timed separately.
''')
cell('code', '''
report, answers = None, []
if RUN_LIVE:
    if not DATABASE.is_file():
        raise RuntimeError('Complete AniList setup before live evaluation. Offline dataset inspection works without it.')
    command = [str(PROJECT_PYTHON), '-u', '-X', 'utf8', '-m', 'src.factual_evaluation',
               '--settings', str(ROOT / 'settings.toml'),
               '--questions', str(RUN_QUESTIONS), '--output', str(OUTPUT)]
    if EXTRACTOR_MODEL:
        command.extend(['--extractor-model', EXTRACTOR_MODEL])
    if MAX_NEW_QUESTIONS is not None:
        command.extend(['--max-new-questions', str(MAX_NEW_QUESTIONS)])
    # Use the managed environment, never the notebook kernel's globally installed RAG dependencies.
    environment = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    if returncode:
        raise RuntimeError(f'Evaluation failed with exit code {returncode}; see the output above. Completed checkpoints are preserved.')
    report = json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8'))
elif (OUTPUT / 'summary.json').is_file():
    saved = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['questions_sha256'] == hashlib.sha256(RUN_QUESTIONS.read_bytes()).hexdigest(), 'Saved dataset differs.'
    assert saved['question_ids'] == [q['question_id'] for q in selected], 'Saved selection differs.'
    report = json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8'))
    print('Displaying saved results; no live calls made during this notebook execution.')
else:
    print('No saved run. Set RUN_LIVE = True to collect actual RAG answers.')
if report is not None:
    answers = read_csv(OUTPUT / 'answers.csv')
    print(f"Actual attempts: {report['overall']['attempted']} / {len(selected)} selected / 180 total prompts")
    if MODE == 'smoke':
        print('SMOKE ONLY: not the 180-question benchmark score.')
    display(pd.DataFrame([report['overall']]).drop(columns=['status_counts']))
    print('Statuses:', report['overall']['status_counts'])
    display(pd.DataFrame([{'field': field, **metrics} for field, metrics in report['by_field'].items()]))
''')
cell('markdown', '''
## Evaluation visualizations

Charts use the selected run's saved answers. Accuracy includes every attempted question;
value precision/recall/F1 include scored answers only. Missing scores remain unscored.
Outcome counts distinguish incorrect answers, unanswered questions, model errors,
extraction errors, and uncertain extractions. Latency includes failures and startup.
Every result chart reports attempted/total benchmark questions and labels partial runs.
''')
cell('code', '''
if report is not None and answers:
    result_charts = show_evaluation_charts(OUTPUT)
else:
    print('No evaluated answers yet. Result charts will appear after collecting or loading a run.')
''')
cell('markdown', '''
## Error analysis and extraction audit

Review failures and apparently correct answers. Retaining the original answer does not guarantee
correct extraction or entity association. A small model judging its own answers can
introduce correlated errors. Exact matching can reject unregistered valid aliases;
review and document alias changes rather than changing expected answers to fit outputs.
''')
cell('code', '''
if answers:
    answer_frame = pd.DataFrame(answers)
    failed = answer_frame.loc[answer_frame.exact_match.ne('1')]
    display(failed[['question_id', 'field', 'question', 'answer_text', 'expected_values',
                    'extracted_values', 'status', 'error']])
    for row in answer_frame.groupby('field', sort=False).head(1).to_dict('records'):
        print('\\n', row['question_id'], row['question'])
        print('ANSWER:', row['answer_text'])
        print('EXPECTED:', row['expected_values'], '| EXTRACTED:', row['extracted_values'])
        print('EXTRACTION AND QUOTES:', row['extraction'])
    print('Full answers:', OUTPUT / 'answers.csv')
else:
    print('Run an experiment before inspecting model errors.')
''')
cell('markdown', '''
## Offline scoring sanity checks

These are synthetic correctness checks on the scorer, not RAG performance measurements.
''')
cell('code', '''
assert score_values([24], ['24.0'], 'episodes')['exact_match'] == 1
assert score_values(['MANGA'], ['manga'], 'source_material')['exact_match'] == 1
assert score_values(['Studio Pierrot'], ['Pierrot'], 'main_animation_studios')['exact_match'] == 1
assert score_values([24], [12], 'episodes')['exact_match'] == 0
partial = score_values(['A', 'B'], ['A'], 'main_characters')
assert partial['precision'] == 1 and partial['recall'] == 0.5
extra = score_values(['A'], ['A', 'Wrong'], 'main_characters')
assert extra['precision'] == 0.5 and extra['recall'] == 1
print('Offline checks passed. Synthetic scores are not reported as model accuracy.')
''')
cell('markdown', '''
## Reproducibility and full-run command

Each run saves final prose, extracted values, quotes, timings, errors, per-field scores,
and per-question JSON debug checkpoints. The dataset manifest freezes catalog metadata,
question hash, and label hash. A run manifest freezes settings and extractor version.
Use a new run name after changes to pipeline code, prompts, or normalization: the run
manifest does not fingerprint all source files. Never silently refresh benchmark labels.

From the repository root, alternatively run all 180 prompts with:

```powershell
.venv/Scripts/python.exe -m src.factual_evaluation --questions data/evaluation/hard_facts/questions.csv --output data/evaluation/hard_facts/results/full_001
```
''')

notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
    'kernelspec': {'display_name': 'Python 3 (project environment)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'},
})
path = ROOT / 'notebook/hard_facts_evaluation.ipynb'
write_notebook(path, notebook)
print('Created', path)
