# Natural-language recommendation benchmark

Open [`notebook/recommendation_evaluation.ipynb`](../../../notebook/recommendation_evaluation.ipynb).
It validates the references, shows coverage, runs the real RAG on demand, displays
saved scores, and audits every recommended title. Live inference is off by default.

The dataset has **72 individually authored English requests**, including **48 indirect
phrasings**. These are synthetic examples of natural user questions, not real user logs.
No external prompt dataset or model-generated gold recommendations were used.

Git contains the small question files, semantic policy, manifests, and notebooks
with their saved inline outputs. The large `catalog_reference.jsonl`, prepared
knowledge bases, exported charts, and separate model-run results are local artifacts and are not
pushed. After AniList setup, run `scripts/build_recommendation_benchmark.py` to
create the local reference and intentionally align labels with that snapshot.
Both coverage plots and the notebook's offline inspection run without that reference.
The implication section uses real catalog records only; when unavailable it reports
the setup prerequisite without substituting illustrative records or model performance.
Reference-dependent tests skip when the local reference is absent; source-input
integrity and metric unit tests still run without it. New-response scoring always
requires the complete local reference, and saved model-run results are never fabricated.

| Category | Questions |
| --- | ---: |
| Tags and mood proxies | 21 |
| Multiple required tags | 3 |
| Required and excluded tags | 3 |
| Genres | 12 |
| Release years | 6 |
| Full release dates | 3 |
| Main animation studios | 6 |
| Release formats | 6 |
| Adaptation source | 3 |
| Episode count / episode runtime | 6 |
| Mixed genre, year, and source constraints | 3 |

Examples include wanting a comforting watch after work, stories that will make the
viewer cry, investigators solving cases, animation by the studio behind another
named title, releases from the nineties, films rather than series, and episodes that
fit into a short break. Desired recommendation counts range from three to five.

## Files and provenance

- `questions.jsonl`: canonical requests, target counts, expected constraints, label
  rationales, category, wording, and a few illustrative catalog matches.
- `questions.csv`: the same data in an inspection-friendly table. Structured cells
  contain JSON; `reference_examples` are examples, not an exhaustive acceptable list.
- `catalog_reference.jsonl`: compact frozen references for all 20,428 local catalog
  entries, including canonical title, aliases, all 422 available catalog tag ranks, genres, year,
  complete start date, main animation studios, format, source medium, and lengths.
- `dataset_manifest.json`: version, source snapshot identifiers, coverage, and SHA-256
  hashes of the question files, catalog, and semantic policy. Every evaluation validates these hashes.
- `semantic_policy.json`: reviewed, request-specific alternative evidence paths and
  their rationales. Edit the source `scripts/recommendation_semantic_policy.json`
  and rebuild intentionally to revise these annotations.
- `results/<run>/`: request checkpoints, run manifest, answers, metric summaries,
  recommendation-level CSV, and charts.

References are queried **read-only from the imported AniList SQLite snapshot**, including
`anime_metadata.search_data.tag_details`. The local snapshot, rather than today's
live website, is the authority for this benchmark. Full dates require known year,
month, and day. Studios require both `main` and `animation` flags. Episode count and
year refer to the exact catalog entry, not the combined franchise. Runtime means
minutes per episode. No online data refresh occurs during evaluation.

To intentionally refresh the dataset and its hashes:

```powershell
.venv/Scripts/python.exe scripts/build_recommendation_benchmark.py
```

Review and version the resulting reference changes. Existing runs cannot resume
against changed references or settings; choose a new run directory.

## Extraction and identity resolution

Only the question text reaches `RagPipeline.ask`. History and agent state are reset
for every request. A second Ollama call receives only the question and final answer,
with a constrained JSON schema, to extract positively recommended titles in order.
It receives no expected constraints, target count, reference examples, or catalog.
The parser excludes comparisons and rejected suggestions, retains duplicate list
entries, and returns one written alias when an item lists multiple names.

Invalid JSON/schema or a title absent from the answer triggers one retry. Unclear
extraction is retained for review, not quietly scored as a normal answer. LLM parsing
can still omit or misclassify titles; schema validation does **not** guarantee semantic
correctness. Saved answers and complete extractions support manual review. Set
`EXTRACTOR_MODEL` to an independently installed stronger model if available.

Resolution uses normalized exact catalog aliases, preserving season identifiers.
Normalization removes punctuation and differences in case/Unicode width. Unknown
and ambiguous titles remain unresolved and receive zero credit. There is no fuzzy
fallback that guesses a season, translates a title, or silently chooses one duplicate
alias. Repeated aliases for the same catalog ID count as duplicate recommendations.

## Constraints and metrics

Each request has an independently authored constraint list:

```json
{"field":"tags","op":"any","value":["Iyashikei"]}
{"field":"genres","op":"any","value":["Fantasy"]}
{"field":"year","op":"gte","value":2020}
{"field":"studios","op":"any","value":["Kyoto Animation"]}
```

All constraints are required for strict success. `any` accepts an alternative;
`all` requires every value; `none` excludes all named values. Scalar attributes use
`gte`, `lte`, or `eq`, with inclusive date/year boundaries. Missing scalar values
score zero. Expected tag names are canonical AniList names, not invented mood labels.
Each request includes its annotation rationale: for example, comforting daily-life
viewing uses Iyashikei, and a heartbreaking story with unhappy outcomes uses Tragedy.

### Accepting reasonable implications

Version 2 gives **32 requests** additional reviewed evidence paths. A
clue-solving request accepts **Detective tag OR Mystery genre**. A broad comedy
request also accepts prominent Slapstick, Surreal Comedy, or Parody tags. Horror
requests accept Body Horror or Cosmic Horror; Gore alone does not qualify. Found-family
wording can be supported by Adoption. Other alternatives require combinations: for
example, the harsh technological future request accepts Sci-Fi **and** Dystopian
**and** either Cyborg or Artificial Intelligence. Every condition in a path must hold
for the same recommended anime. The stored rationale explains each decision.

These are directional, request-specific annotations. They are not universal synonym
rules: a Mystery anime need not literally have a Detective tag. The request that
explicitly says "carry the Detective tag" retains its exact requirement; the broader
investigation requests accept Mystery. Likewise, an unhappy-ending request is not
automatically replaced by generic Drama. Exact release dates, studios, length limits,
and other required conditions remain in effect, including in mixed requests.

An intent's graded score is the maximum of its direct-label score and all approved
alternative-path scores. A path's score is the minimum of its supporting condition
scores. An intent passes if any path satisfies all its conditions, respecting the
strict exclusion threshold. Different intents remain jointly required. Alternative
paths are evidence choices, not additional requirements or extra metric weight.
Unrequested extra tags/genres never incur a penalty. All available catalog tag ranks
are frozen, so scoring does not drop non-target labels from the reference snapshot.

`constraint_evidence` in each recommendation audit records the literal score,
accepted score, credited path, supporting conditions, and rationale. **Literal P@k**
is reported alongside intent-aware P@k. **Semantic credit@k** is the fraction of slots
that fail literal matching but pass through an accepted implication; the two sum to
intent-aware P@k. The notebook includes a clearly labeled reference-only demonstration
using real Mystery titles that lack a strong Detective tag, plus a comparison chart.
Policy coverage remains authored and finite; review new ambiguous intents and add
defensible paths rather than assuming all related themes are interchangeable.

For tag constraints, the score is the catalog relevance rank / 100. `any` takes the
maximum, `all` the minimum, and `none` one minus the maximum. Non-tag constraints
score 0 or 1. The recommendation's graded gain is the equal-weight mean across
constraints. A strict match requires all positive tag scores >= 0.5, excluded tag
ranks < 50, and every non-tag constraint to pass. For `all` tag groups, the minimum
rank must reach 50. This threshold remains fixed at 50 in version 2.

For requested count k and returned order:

| Metric | Computation / interpretation |
| --- | --- |
| Strict P@k | Fully compliant unique resolved recommendations in the first k positions / k |
| Literal P@k | P@k using only direct labels, without implication paths |
| Semantic credit@k | Fraction of slots rescued by approved alternatives |
| Graded fit@k | Sum of graded gains in the first k positions / k |
| nDCG@k | Sum(gain_i / log2(i+1)), positions starting at 1, divided by the ideal DCG from the top k unique entries in the entire reference catalog |
| Constraint coverage | Fraction of constraints met by at least one recommendation in the first k; not necessarily the same recommendation |
| Count compliance | Exactly k extracted recommendations |
| Resolution rate | Resolved extracted recommendations / all extracted recommendations |
| Duplicate rate | Repeated identities or repeated unresolved names / all extracted recommendations |
| Extra count | max(returned count - k, 0) |

Missing slots, unresolved identities, and duplicates receive zero ranking gain and
strict credit. Items beyond k remain in the audit and extra/count metrics, but not
top-k quality metrics. An empty list scores zero; empty resolution/duplicate rates
are zero and must be interpreted alongside returned counts.

Metrics are macro-averaged per request so long lists do not dominate. Primary
**end-to-end P@k** includes every attempted request, with model failures and unclear
or failed extraction contributing zero. **Scored-only averages** exclude unscored
errors and are always paired with **scoring coverage**. Unattempted questions are
pending, not failures or successes. An all-error scored-only group has null metrics.
No score is reported for an unattempted run.

`recommendations.csv` preserves each item's individual constraint scores, raw graded
fit, duplicate-adjusted gain, title candidates, and strict result. `answers.csv`
preserves the question, original answer, extraction, errors, timings, and metrics.

## Run and resume

Install notebook/chart dependencies with `uv sync --extra evaluation`. Prepared
AniList indexes and the configured Ollama model must already be available.

```powershell
# All 72; use an unused run name.
.venv/Scripts/python.exe -m src.recommendation_evaluation --mode full --output data/evaluation/recommendations/results/full_001

# Limit new work in an existing identical run without changing the selection.
.venv/Scripts/python.exe -m src.recommendation_evaluation --mode full --max-new-questions 5 --output data/evaluation/recommendations/results/full_001
```

`--extractor-model <installed-name>` changes the extraction model; this requires a
new output directory. Saved failures are also resumed; use a new run name to retry.
`--max-new-questions 0` loads only existing checkpoints and never initializes the RAG.

The notebook uses the project's `.venv` for generation and charts even if Jupyter
uses another kernel. The notebook selects all 72 requests and rejects subset runs.
Set `RUN_LIVE=True` with a new name for fresh full inference.
Default offline mode shows the saved `full_real_v2` run when present. Chart
exports include coverage, strict compliance and scoring coverage, quality breakdowns,
outcomes and title resolution, a per-item constraint heatmap, and measured latency.
PNG and SVG files are saved alongside their inline notebook displays.

`full_real_v2` is a complete 72-request model run: all requests were attempted,
61 received automatic scores, 10 had generation errors, and one required extraction
review. It reuses genuine full-run answers and LLM title extractions, deterministically
rescored against the current reference and implication policy. The original `change`
run remains intact and source hashes are saved in `rescoring_provenance.json`.
These are historical model responses, not freshly generated answers from every
subsequent code revision. A new live run is required to measure a new pipeline.

To apply revised labels to saved answers without calling the RAG or extractor:

```powershell
.venv/Scripts/python.exe -m src.recommendation_evaluation --rescore-from data/evaluation/recommendations/results/change --output data/evaluation/recommendations/results/my_semantic_rescore
```

Use a new destination. Question text and desired counts must match. The new run
retains generation/extraction timing, previous scores, and source checkpoint hashes
in `rescoring_provenance.json`. Old manifests cannot resume as version-2 live runs;
rescore into a new run or start a fresh one.

```powershell
.venv/Scripts/python.exe -m unittest tests.test_recommendation_evaluation -v
.venv/Scripts/python.exe scripts/verify_recommendation_notebook.py
```

The verifier executes every notebook cell in a real project kernel with live inference
disabled, validates the notebook, and saves verified tables and inline charts.

### Full-date request fix

The historical smoke run preserves the original `dates_01` error. The application
now parses explicitly bounded YYYY-MM-DD release dates into `start_date` calendar
filters. Month/day precision is retained; unknown or invalid catalog dates do not
match. Inclusive from/through ranges and exclusive before/after comparisons are
supported. Existing year-only and decade requests keep their numeric resolver.
The planner cannot replace validated date bounds with malformed numeric year filters.

`results/date_fix_validation/` contains fresh live answers for the formerly failing
full-date request and an existing year-only control. Both were scored with strict
P@k = 1, without tool errors. This is a focused fix check, not a fresh full benchmark.
The previously observed extra-recommendation count remains visible in those metrics.

To repeat the live check, choose a new output directory:

```powershell
.venv/Scripts/python.exe scripts/verify_recommendation_date_fix.py --output data/evaluation/recommendations/results/date_fix_check_002
```

## Limits

This measures catalog constraint adherence, not enjoyment, taste, novelty, or overall
recommendation quality. Mood proxies and community tag ranks are subjective. An absent
tag means no positive evidence in the snapshot; it cannot prove absence. Graded mean
fit can give partial credit to a recommendation that fails a required constraint;
strict P@k exposes that failure. Constraint coverage can be high even if different
items satisfy incompatible parts of a request. nDCG is normalized against the local
catalog and is not recall over all anime or a subjective ranking oracle.

The requests are synthetic, in English, and skew toward tags/genres. The explicit and
indirect sets request different counts (3 versus 4/5), so their breakdown is descriptive,
not a controlled wording experiment. Personalization, multi-turn behavior, real traffic
frequency, precise franchise-wide runtime, and unknown/unanswerable constraints are
outside this version. Report sample sizes, attempted/planned counts, and extraction
coverage with every performance claim. The notebook's selected evaluation includes
all 72 requests, with failures retained in end-to-end accuracy denominators.
