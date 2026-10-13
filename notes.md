# Notes: MultiDoc2Dial dataset

Adds a public, annotated dataset so we can index real documents and measure retrieval quality automatically.

## What's new

One new command:

```sh
uv run rag prepare-multidoc2dial
```

It does three things:

1. Downloads the official MultiDoc2Dial zip (6.9 MB) from `doc2dial.github.io`.
2. Writes its **488 documents** as Markdown files to `data/documents/multidoc2dial/<domain>/<name>.md`. The domains are `dmv` (NY driving licences), `ssa` (social security), `va` (veterans' benefits) and `studentaid` (US student financial aid). Each file holds the page title and its plain text.
3. Writes **3,423 evaluation questions** from the validation split to `data/evaluation/multidoc2dial_validation.jsonl`.

After that, `uv run rag ingest` indexes the documents as usual. Nothing else in the pipeline changes.

## Files changed

| File | Change |
|---|---|
| `src/multidoc2dial.py` | New. Download, document export, and question extraction. |
| `src/cli.py` | Registers the `prepare-multidoc2dial` command. `ingest` and `ask` behave as before. |
| `.gitignore` | Ignores the generated data (`data/documents/multidoc2dial/`, `data/evaluation/`). Run the command to regenerate it. |

## Evaluation file format

One JSON object per line, for each user turn that asks something (dialogue acts `query_condition` and `query_solution`):

| Field | Meaning |
|---|---|
| `question` | The user utterance. |
| `history` | Previous turns of the same dialogue (`"user: ..."`, `"agent: ..."`). Many questions only make sense with this context. |
| `gold_source` | Path of the document that answers the question, relative to `data/documents`. Same format as `source` in `data/index/chunks.json`, so retrieval hits can be compared directly. |
| `gold_doc_id`, `dialogue_id`, `turn_id`, `domain` | Original dataset identifiers. |

## Why this dataset

- Every question is annotated with the document that answers it, so retrieval can be scored automatically (recall@k) instead of writing test questions by hand.
- The paper reports baselines to compare against: BM25 finds the right passage in the top 5 for 42.7% of validation questions; a fine-tuned dense retriever reaches 72.3%.
- License: Apache-2.0. Citation: Feng et al., *MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents*, EMNLP 2021.

## Known limitations

- The documents are a snapshot from about 2020; some content is outdated.
- Some turns labelled as questions are not real questions (for example "Don't do that I'll get insurance"). Filter them in the evaluation step.
- The dataset has no unanswerable questions, so testing refusals needs a small hand-written set.
- The test split in the zip is a placeholder; use the validation split.

## To do

Deadlines: code + report + contribution statements **Oct 20**, demo **Oct 22-23**.

**Team decisions**
- [ ] Confirm MultiDoc2Dial (or pick another dataset) and split the tasks below.
- [ ] `data/index/` is tracked in git. With 488 documents the index grows to several MB and changes on every ingest. Decide whether to ignore it too.

**Evaluation (no Ollama needed)**
- [ ] `rag evaluate`: read the evaluation file, retrieve for each question, report recall@1/5/10 (gold document among the top k chunks' sources).
- [ ] Drop turns that are not real questions (for example "Don't do that I'll get insurance").
- [ ] Baseline numbers with the current `settings.toml`.

**Retrieval experiments (report: design choices backed by experiments)**
- [ ] Chunk size and overlap.
- [ ] `top_k`.
- [ ] Embedding model: `all-MiniLM-L6-v2` vs one stronger small model.
- [ ] Query built from the question only, question + dialogue history, or a question rewritten by the LLM.
- [ ] Optional: hybrid search (BM25 + embeddings) or a cross-encoder reranker.

**Answers (needs Ollama + `llama3.2:3b`)**
- [ ] Answer quality on a sample of 200-300 questions, compared with the reference agent replies.
- [ ] Citations: check that the cited file is the gold document.
- [ ] Refusals: about 30 hand-written off-topic or unanswerable questions; measure how often the bot correctly says it does not know. The challenger groups will try exactly this at the demo.

**Demo**
- [ ] GUI shows the sources of each answer and keeps the conversation history.
- [ ] Latency check (streaming if answers are slow).
- [ ] A short list of good example questions per domain.

**Report (4-6 pages)**
- [ ] System design and implementation.
- [ ] Experiment results tables.
- [ ] Societal impact: outdated government information, the risk of wrong benefits advice, accessibility of public services, accountability.
- [ ] Declaration of generative AI use.
- [ ] Each member: individual contribution statement.
