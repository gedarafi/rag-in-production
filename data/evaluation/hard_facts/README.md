# Hard-fact natural-language benchmark

Open `notebook/hard_facts_evaluation.ipynb` from the repository root or notebook directory.
The main dataset is `questions.csv`: **180 individually authored English prompts across
76 anime**, with no repeated question text or anime/field pair. These are realistic
synthetic questions, not collected user questions. No external question dataset was used.

| Requested fact | Questions |
| --- | ---: |
| First release year | 26 |
| Episode count | 26 |
| Minutes per episode | 26 |
| Release format | 26 |
| Source medium | 26 |
| Main animation studios | 25 |
| Main characters | 25 |

Questions deliberately mix direct questions, informal requests, season disambiguation,
film/TV distinctions, and complete-list requests. Popular titles are overrepresented;
this is a reproducible initial benchmark, not a representative sample of user traffic.
Several titles occur under different fields; each anime/field target occurs once.
Release dates are evaluated at **year precision only**. All questions have a nonempty
catalog answer; unknown-answer handling and multi-turn behavior are outside this set.

## Frozen reference labels

`expected_values` is a JSON array stored in a CSV cell. IDs are AniList IDs.
Labels come directly from `data/documents/anilist/catalog.sqlite3`, queried read-only.
Studios require both the main and animation flags; characters require role `MAIN`.
Year/episodes/duration reflect the specific entry, not the combined franchise.
`dataset_manifest.json` records the catalog metadata, dataset hash, and label hash.
The source is the imported local snapshot, not today's live AniList website.

The notebook checks hashes and displays dataset coverage without a local database.
When the database exists, it checks every label against it; live inference always
requires that database check. If labels
drift, investigate the source change rather than silently replacing expected answers.
To intentionally create refreshed labels, run from the root:

```powershell
.venv/Scripts/python.exe scripts/build_hard_facts_benchmark.py
```

## Evaluation

The notebook uses `src.factual_evaluation.run_evaluation`: actual `RagPipeline.ask`
responses, empty history and fresh agent state per question, followed by automatic
claim extraction and deterministic value comparisons. Only question text is sent as
the user request; expected values, entity IDs, and requested-field annotations are
not supplied to the RAG. The extractor receives the question, field, and response,
but never reference values. A dedicated LLM request parses values; code attaches the full original answer as evidence.

Set `RUN_LIVE = True` in the notebook to run inference. Choose `MODE = 'full'` for
all 180 questions or `'smoke'` for seven questions covering all fields. The smoke
CSV is deterministically selected by the notebook from the main dataset. Ollama,
the configured model, and prepared AniList indexes must be available. No setup or
dataset rebuilding is performed. Results resume from per-question checkpoints;
choose a new `RUN_NAME` to retry failures or compare another configuration.

Live inference runs in a subprocess using the repository's `.venv` Python, rather
than importing the RAG's dependencies into the notebook kernel. This allows a
Python 3.11+ notebook kernel to use the managed Python 3.14 project environment
and avoids conflicting global `docx` installations. The notebook kernel itself
needs pandas. Console progress streams into the notebook; interrupting stops the
evaluation subprocess and preserves completed checkpoints.

Install visualization dependencies with `uv sync --extra evaluation`. The notebook
includes dataset coverage, per-field exact accuracy versus scoring coverage, answer
outcomes, scored-answer value precision/recall/F1, and mean response/extraction latency.
Charts render in the project environment and display inline. PNG and SVG exports
are saved in `charts/` for the dataset and `results/<RUN_NAME>/charts/` for each run.
Partial runs are labeled with their attempted-question count; unscored metrics are
never presented as zero accuracy.

Extractor version 3 uses a schema-constrained LLM request for every answer. It returns parsed values without generating quotes; code attaches the entire original answer as evidence. Frozen labels never reach the parser. To repair extraction on existing responses without
rerunning the RAG or modifying the original run:

```powershell
.venv/Scripts/python.exe -m src.factual_evaluation --questions data/evaluation/hard_facts/questions.csv --reextract-from data/evaluation/hard_facts/results/full_001 --output data/evaluation/hard_facts/results/full_001_parser_v3
```

Use a new output directory. The repaired run preserves the original answers,
generation latency, debug events, and prior extraction details. In the notebook,
set `RUN_LIVE = False` and `RUN_NAME` to the repaired directory name to inspect it.
Do not resume an old version-1 run with version-2 extraction; choose a new run name.

Version-2 repaired runs can resume live inference with matching settings. Extra
provenance fields are preserved and do not invalidate an otherwise identical run.
Set `MAX_NEW_QUESTIONS` in the notebook to cap new attempts per execution without
changing the 180-question selection (`None` continues all remaining questions;
`0` loads completed checkpoints only). This differs from `--limit`, which changes
the benchmark selection and therefore requires a separate run directory.

To verify every cell in a real project Jupyter kernel, including one new live attempt:

```powershell
.venv/Scripts/python.exe scripts/verify_hard_facts_notebook.py --max-new-questions 1
```

The verifier temporarily caps that execution, restores the configured budget, and
saves verified outputs only after every cell completes without an error.

Results live in `results/<RUN_NAME>/`: `answers.csv`, per-question JSON checkpoints
including debug events, `summary.json`, and a run configuration manifest. Startup
time is included in the first response latency; extractor time is separate.

Primary metric: **end-to-end exact accuracy across every attempted question**.
Also inspect per-field accuracy, scoring coverage, scored-answer macro value F1,
errors, and latency. Errors and uncertain extractions remain in the primary
denominator. A seven-question smoke result is only an integration check, not the
180-question benchmark score. Automatic extraction can be wrong, especially when
the same small model generates and judges; manually audit the saved answer quotes.
Scores concern the requested field only, not every extra claim in the answer.

## Included live integration run

`results/smoke_validation/` contains seven real RAG attempts (one per field) using
`llama3.2:3b` for generation and extraction. Three answers were exact matches, one
was wrong, and three failed the RAG's grounded-response validation. End-to-end exact
accuracy was **3/7 (42.9%)**, with scoring coverage **4/7**. The incorrect answer
asserted 62 episodes for Hunter x Hunter (2011), versus 148 in the catalog. Saved
quotes and debug checkpoints allow inspection. This is an integration sample only;
**the full 180-question experiment has not been run**. The notebook displays these
saved smoke results by default and makes no live calls until enabled.

Extractor v3 parses every answer with an LLM, including numeric answers. Evidence contains the entire original response, not a selected supporting span. Transport and malformed-output failures remain visible; unclear answers require review.
