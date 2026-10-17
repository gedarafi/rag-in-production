"""Build a clean, runnable recommendation evaluation notebook (standard library)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = {'cell_type': kind, 'id': f'recommendation-{len(cells):02d}', 'metadata': {},
            'source': source.strip().splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell('markdown', '''
# Recommendation evaluation: natural requests and catalog constraints

**72 individually authored naturalistic requests; 48 use indirect wording.**
Topics include mood and tags, genres, release years and full dates, animation studios,
formats, adaptation source, episode counts, episode runtime, and combined constraints.
These are synthetic examples of user questions, **not collected user logs**.

The actual RAG answers each question. A **separate schema-constrained LLM request**
extracts recommended titles in order. Deterministic code resolves title aliases and
scores every recommendation against frozen catalog attributes. Neither the RAG nor
the extractor receives the reference labels. This notebook defaults to offline
inspection and displays saved live results when available.
''')
cell('code', '''
from pathlib import Path
import json
import os
import subprocess
import sys
import pandas as pd
from IPython.display import display, Image

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src/recommendation_evaluation.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook from the repository root or notebook folder.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.recommendation_evaluation import load_dataset, METRICS
from src.config import load_settings

DATA = ROOT / 'data/evaluation/recommendations'
PYTHON = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not PYTHON.is_file():
    raise RuntimeError('Run uv sync --extra evaluation from the repository root first.')
questions, catalog, manifest = load_dataset(DATA)
frame = pd.DataFrame(questions)
SETTINGS = load_settings(ROOT / 'settings.toml', 'anilist')
print(f'{len(questions)} questions | {len(catalog.records)} frozen catalog entries | {frame.category.nunique()} categories')
print('All reference hashes verified. No database mutation, inference, or downloads.')
''')
cell('markdown', '''
## Labels and coverage

Each request specifies its desired count `k` and a list of expected constraints.
Constraints use `any`, `all`, `none`, `gte`, `lte`, or `eq`. All listed constraints
must hold for a strict match. Alternative acceptable tags can share one `any` group.
For example, “something comforting after work” can be supported by **Iyashikei**, or
an annotated combination of everyday-life genre/themes without a strong tragic theme.
An investigation request accepts **Detective OR Mystery**, across tags and genres.
**32 requests have reviewed alternative evidence paths**. Every catalog tag is retained
in the reference (422 in this snapshot), rather than freezing only the original target tags.

Alternative paths are written per request in `semantic_policy.json`, with rationales.
Each path can require several supporting conditions together. Unrelated extra tags
never reduce a score. Precise requests such as “carry the Detective tag” retain that
literal requirement; broader clue-solving requests accept Mystery. Exclusions and
dates, studios, counts, and other independent requirements still apply. These are
reviewable annotation choices, not universal synonym rules or a guarantee of mood.

Indirect studio requests include identifying the studio behind another named anime.
Dates refer to the first release of the specific recommended entry. Studios must be
flagged as both main and animation studios. Missing scalar metadata fails its constraint.
''')
cell('code', '''
display(pd.crosstab(frame.category, frame.wording).assign(total=lambda t: t.sum(axis=1)))
display(frame.groupby('category', sort=False).head(1)[
    ['question_id', 'question', 'k', 'constraints', 'strict_catalog_matches']])
display(frame[frame.wording.eq('indirect')].sample(6, random_state=7)[
    ['question', 'constraints', 'label_rationale']])
assert frame.question_id.is_unique and frame.question.str.casefold().is_unique
assert frame.question_origin.eq('authored_naturalistic').all()
print('Dataset provenance:', manifest['origin'])
''')
cell('markdown', '''
## Check that valid implications receive credit

This reference-only demonstration uses real catalog entries, **not model outputs**.
It finds anime recorded as Mystery without a strong Detective tag and compares the
same entries under an explicit tag request and the broader investigation request.
The latter receives credit through the reviewed Mystery evidence path.
''')
cell('code', '''
lookup = {q['question_id']: q for q in questions}
implication_examples = []
for record in catalog.records.values():
    if 'Mystery' in record['genres'] and record['tag_ranks'].get('Detective', 0) < 50:
        implication_examples.append({
            'title': record['title'], 'Detective rank': record['tag_ranks'].get('Detective', 0),
            'Mystery genre': True,
            'explicit tag request': catalog.fits(lookup['detectives_01'], record)[1],
            'investigation request': catalog.fits(lookup['detectives_02'], record)[1]})
    if len(implication_examples) == 5:
        break
display(pd.DataFrame(implication_examples))
print('Accepted alternatives:', manifest['requests_with_alternatives'], 'requests')
''')
cell('markdown', '''
## Metric definitions

For a tag constraint, the graded value is its catalog relevance rank divided by 100;
`any` takes the maximum, `all` the minimum, and `none` one minus the maximum.
Other constraints score 0 or 1. A recommendation's **graded fit** is their equal-weight
mean. Each intent takes the best score across its literal path and reviewed alternatives;
an alternative's score is the minimum of its supporting condition scores. **Strict match**
requires a passing path for every intent. Positive tag ranks must reach 50, excluded
tags stay below 50, and all non-tag conditions in that path must pass.

| Metric | Definition |
| --- | --- |
| Strict P@k | Unique, resolved, fully compliant recommendations in the first k positions / requested k |
| Literal P@k | The same computation using only the original direct labels, for comparison |
| Semantic credit@k | Fraction of requested slots rescued by an accepted alternative |
| Graded fit@k | Sum of mean constraint fits in those positions / k |
| nDCG@k | Rank-discounted graded fit / ideal fit from the best k unique entries in the entire frozen catalog |
| Constraint coverage | Fraction of expected constraints met by at least one top-k recommendation; different items may meet different constraints |
| Count compliance | Extracted list length equals requested k |
| Resolution / duplicate rates | Fractions across all returned recommendations, including extras |

Missing positions, unknown titles, ambiguous aliases, and duplicate entries contribute
zero to P@k and graded/ranking gains. Recommendations beyond k are still audited and
counted as extras. nDCG uses linear gains and log2 discounts. It measures ranking of
catalog fit, not subjective quality or catalog-recall. End-to-end P@k includes all
attempted requests, with model errors and unclear parses contributing zero. Conditional
scored-only metrics and scoring coverage are shown separately. Unattempted requests
are pending and excluded from performance denominators.
''')
cell('code', '''
def show_charts(results=None):
    output = Path(results) / 'charts' if results is not None else DATA / 'charts'
    command = [str(PYTHON), '-X', 'utf8', '-m', 'src.recommendation_visualizations',
               '--dataset', str(DATA), '--output', str(output)]
    if results is not None:
        command += ['--results', str(results)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', env={**os.environ, 'MPLBACKEND': 'Agg'})
    if completed.returncode:
        raise RuntimeError('Chart rendering failed. Install uv sync --extra evaluation. ' + completed.stderr)
    artifacts = json.loads((output / 'charts.json').read_text(encoding='utf-8'))
    for artifact in artifacts:
        if results is not None and Path(artifact['png']).stem == 'dataset_coverage':
            continue
        display(Image(filename=artifact['png']))
        print(artifact['caption'])
    print('PNG and SVG exports:', output)

show_charts()
''')
cell('markdown', '''
## Run the actual model

Set `RUN_LIVE = True` to evaluate. `MODE = 'smoke'` selects one request per category
(11 requests); `MODE = 'full'` selects all 72. Use a new `RUN_NAME` for a new model,
new extraction model, changed settings, or a retry of a failed question. Identical
runs resume per-question checkpoints, including failures. `MAX_NEW_QUESTIONS` limits
new attempts per execution without changing the selection; `None` continues all.

The prepared AniList indexes and Ollama model must already be available. Live inference
runs in the project's interpreter to isolate it from conflicting notebook dependencies.
Every question starts with empty history and fresh agent state. Extraction transcribes
only positively recommended titles, excluding comparisons and rejected suggestions.
It preserves duplicates and order. Failed JSON validation is retried once; uncertain
extraction is marked for review. LLM extraction cannot guarantee error-free parsing.
''')
cell('code', '''
RUN_LIVE = False
MODE = 'smoke'                  # 'full' for all 72 questions
RUN_NAME = 'smoke_semantic_reviewed'  # change for a new experiment
EXTRACTOR_MODEL = None          # e.g. a separately installed stronger Ollama model
MAX_NEW_QUESTIONS = None        # 0 loads checkpoints only; None continues remaining requests

if MODE not in {'smoke', 'full'}:
    raise ValueError('MODE must be smoke or full')
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {'.', '..'}:
    raise ValueError('RUN_NAME must be a single directory name')
OUTPUT = DATA / 'results' / RUN_NAME
print('Generation:', SETTINGS.llm, '| Extraction:', EXTRACTOR_MODEL or SETTINGS.llm)
print('Mode:', MODE, '| Live inference:', RUN_LIVE, '| Output:', OUTPUT)
''')
cell('code', '''
if RUN_LIVE:
    command = [str(PYTHON), '-u', '-X', 'utf8', '-m', 'src.recommendation_evaluation',
               '--settings', str(ROOT / 'settings.toml'), '--dataset', str(DATA),
               '--output', str(OUTPUT), '--mode', MODE]
    if EXTRACTOR_MODEL:
        command += ['--extractor-model', EXTRACTOR_MODEL]
    if MAX_NEW_QUESTIONS is not None:
        command += ['--max-new-questions', str(MAX_NEW_QUESTIONS)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace',
                               env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Evaluation failed. See subprocess output above.')
    except BaseException:
        process.terminate()
        process.wait()
        raise
else:
    print('Offline inspection. Set RUN_LIVE = True to generate new answers.')
''')
cell('markdown', '''
## Saved results and recommendation-level audit

Partial results are explicitly labeled. The audit table includes every extracted
recommendation, its canonical identity, resolution status, duplicate flag, and each
constraint score. Inspect the original answer alongside extraction whenever a title
is missing, ambiguous, or unexpectedly included. Reference examples in the dataset
are illustrative, not an exhaustive acceptable recommendation list.
''')
cell('code', '''
rows = []
if (OUTPUT / 'answers.json').is_file():
    report = json.loads((OUTPUT / 'metrics.json').read_text(encoding='utf-8'))
    rows = json.loads((OUTPUT / 'answers.json').read_text(encoding='utf-8'))
    overall = report['overall']
    print(f"Attempted {overall['attempted']}/{overall['planned']} selected requests; {overall['pending']} pending.")
    print(f"Full benchmark has {len(questions)} requests. This is {'a full benchmark' if len(rows) == len(questions) else 'a partial benchmark' }.")
    display(pd.DataFrame([{k: v for k, v in overall.items() if k not in {'scored_macro', 'status_counts'}}]))
    display(pd.DataFrame([overall['scored_macro']]).rename(index={0: 'scored-only macro'}))
    summaries = []
    for category, summary in report['by_category'].items():
        if summary['attempted']:
            summaries.append({'category': category, 'n': summary['attempted'],
                              'end_to_end_P@k': summary['end_to_end_precision_at_k'],
                              'scoring_coverage': summary['scoring_coverage'], **summary['scored_macro']})
    display(pd.DataFrame(summaries))
    print('The saved smoke uses explicit requests; the reference-only example above separately exercises an indirect request.')
    audit = [{**d, 'question_id': r['question_id']} for r in rows for d in r['recommendations']]
    if audit:
        display(pd.DataFrame(audit)[['question_id', 'rank', 'extracted_title', 'canonical_title',
                                     'resolution', 'duplicate', 'constraint_scores', 'raw_fit', 'strict_match']])
        evidence_audit = [{'question_id': r['question_id'], 'rank': d['rank'],
                           'title': d['extracted_title'], 'constraint_index': i,
                           'literal_score': e['literal_score'], 'accepted_score': e['score'],
                           'credited_path': e['credited_path'], 'paths': e['paths']}
                          for r in rows for d in r['recommendations']
                          for i, e in enumerate(d.get('constraint_evidence', []))]
        display(pd.DataFrame(evidence_audit).head(15))
    else:
        print('No resolved extraction audits yet. Inspect errors below.')
else:
    print('No saved run yet. Dataset coverage is available above; model scores require RUN_LIVE = True.')
''')
cell('code', '''
if rows:
    show_charts(OUTPUT)
''')
cell('code', '''
# Change this ID to inspect any saved response and its complete extraction.
INSPECT_ID = rows[0]['question_id'] if rows else None
for row in rows:
    if row['question_id'] == INSPECT_ID:
        print('REQUEST:', row['question'])
        print('EXPECTED CONSTRAINTS:', json.dumps(row['constraints'], ensure_ascii=False))
        print('STATUS:', row['status'], row['error'])
        print('FINAL ANSWER:\\n' + row['answer_text'])
        print('LLM EXTRACTION:', json.dumps(row['extraction'], indent=2, ensure_ascii=False))
        break
failures = [{k: r[k] for k in ('question_id', 'status', 'error')} for r in rows
            if r['status'] in {'needs_review', 'extraction_error', 'model_error'}]
if failures:
    display(pd.DataFrame(failures))
''')
cell('markdown', '''
## Interpretation and limits

This benchmark measures adherence to the stated catalog constraints, not whether a
recommendation is enjoyable. Mood-to-tag mappings are human-authored proxies. Tag ranks
reflect the imported community snapshot, not probabilities. Absent tags are zero
positive evidence and do not prove a theme is absent. Titles and season variants can
share aliases; ambiguous matches receive zero credit and remain visible for review.
Studio aliases are fixed through annotation rather than guessed at scoring time.

The prompts are small, synthetic, and in English; genre and tag requests dominate.
Explicit versus indirect comparisons also differ in requested count (3 versus 4/5),
so they are descriptive breakdowns rather than a controlled causal experiment.
The benchmark contains no personalization, real traffic sampling, conversational memory,
or full-franchise runtime evaluation. The extraction model can omit or misidentify
titles; review saved answer/extraction pairs, and use a stronger independent extractor
when available. Report sample sizes and scoring coverage with every quality score.

See `data/evaluation/recommendations/README.md` for file formats, metric formulas,
reproduction commands, and frozen-reference provenance.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python', 'version': '3.14.0'}},
            'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'notebook/recommendation_evaluation.ipynb'
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(path)
