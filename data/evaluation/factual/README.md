# Factual evaluation samples

The published [hard-facts notebook](../../../notebook/hard_facts_evaluation.ipynb)
provides the larger 180-question workflow, with offline inspection and optional live
Ollama evaluation. The 42-question dataset in this directory remains available through
the CLI below and as small synthetic scorer fixtures. Live inference requires the
project environment, completed AniList setup, and an installed Ollama model.

| CSV | Purpose |
| --- | --- |
| `questions.csv` | 42 questions, anime IDs, requested fields, and frozen database values. |
| `answers_correct.csv` | Synthetic fully correct answers; expected accuracy 100%. |
| `answers_mixed.csv` | Synthetic errors, incomplete lists, an extra studio, aliases, and an unanswered question; expected exact accuracy 85.7% (36/42). |
| `answers_template.csv` | Blank response and annotation rows for a real experiment. |
| `answers_real_sample.csv` | Eight actual RAG attempts with automatic extraction, quotes, scores and errors from the live validation run. |

The correct/mixed answer fixtures are synthetic; `answers_real_sample.csv` contains real attempts.
Questions cover release year, episode count,
minutes per episode, release format, source material, main animation studios, and all main
characters for Naruto, Naruto: Shippuden, Clannad, Cowboy Bebop, Death Note, and A Silent Voice.

The eight-question live smoke test scored 5/8 end-to-end exact correctness: five correct answers,
one answer incorrectly including production companies as main animation studios, and two grounded-response
validation failures inside the RAG. Six answers were automatically scored (5/6 exact); both inference
failures remain in the all-attempts denominator. This is a small validation sample, not an estimate
of performance over the full benchmark. It used `llama3.2:3b` for both RAG generation and extraction.

## Real experiment

For this 42-question dataset, use the CLI from the repository root:

```powershell
.venv/Scripts/python.exe -m src.factual_evaluation --output data/evaluation/factual/results/my_run
```

For a short smoke test, add `--limit 3`. An optional `--extractor-model MODEL` selects an installed
Ollama model for extraction; by default the configured RAG model is reused. A separate stronger
model can help reduce correlated mistakes. The extractor never receives expected values or catalog records.

The pipeline calls the actual `RagPipeline.ask` with empty history and fresh agent state per question.
It extracts asserted facts, preserving errors and verbatim supporting quotes, then uses deterministic
set comparisons against `questions.csv`. No setup rebuild is invoked by the evaluation command itself.

Rerunning the identical command resumes saved checkpoints, including failed questions. Choose a new
output directory to retry failures, change the benchmark/settings/limit, or make an independent repeat.
The manifest rejects changes to a run's configuration. Startup is included in the first response timing;
extractor latency is measured separately and is not part of RAG response latency.

Example CSV cell values: `[220]`, `["MANGA"]`, `["Studio Pierrot"]`.
The manual template remains useful for validating extraction: paste real responses, annotate all asserted
values (including errors), set `answer_status=answered` or `unanswered`, and `reviewed=true`.
Use a CSV-aware editor or Python's `csv` module; JSON quotes inside CSV cells need CSV escaping.

The pipeline verifies frozen labels against read-only database queries before inference. It saves:

- `answers.csv`: actual final answers, extracted values, quotes, metrics, timings, and errors.
- `fact_*.json`: resumable per-question checkpoints, including RAG usage and debug events.
- `summary.json`: overall and per-field scores and status counts.
- `manifest.json`: question-file hash, settings, selected question IDs, and extractor choice/version.

End-to-end exact accuracy includes every attempted question: model errors, extraction errors and unclear
extractions are unsuccessful attempts. Scoring coverage and accuracy among scored answers are reported
separately. Macro value precision/recall/F1 apply to scored questions; unanswered responses score zero.

Automatic extraction is fallible, even with valid JSON and verbatim quotes. Audit a sample manually;
a quote's presence does not prove that the extracted value or entity association is correct. Ambiguous
extractions are flagged for review, and operational errors remain visible instead of disappearing
from the end-to-end score. This evaluates requested fields, not all prose claims or recommendations.
Main animation studios are restricted to
records with both `main=true` and `animation=true`; characters are restricted to role `MAIN`.
Database absence is not proof that a statement is false in the real world.
