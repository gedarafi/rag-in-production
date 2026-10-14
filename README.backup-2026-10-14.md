# Team rules

* Create a pull request from a new branch for each contribution.
  * Never push directly to main.
  * It requires at least one approval from another team member to merge.

* Feel free to use AI for anything, it is especially useful for code review.

* We embrace experimentation and collaboration; any idea is welcome.


# GenAI Used so far:

## GitHub Copilot

* Pull request review.

## Claude

* `/code-review --fix`

## Local RAG

This repository contains a configurable Retrieval-Augmented Generation application for local
documents. It uses Sentence Transformers for embeddings and Ollama for generation.

### Setup

Requires Python 3.14 or newer, `uv`, and [Ollama](https://ollama.com/).

#### Windows (PowerShell)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Ollama](https://ollama.com/download/windows), then start Ollama. From the repository root:

```powershell
uv sync
ollama pull llama3.2:3b
```

#### macOS (Terminal)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart Terminal so `uv` is available. Download [Ollama for macOS](https://ollama.com/download/mac),
install it in Applications, and launch it. Complete its first-run setup to enable the `ollama` command.
Keep Ollama running while using the app.

From the repository root, install Python and dependencies, download the model, and launch the GUI:

```bash
uv python install 3.14
uv sync
ollama pull llama3.2:3b
uv run python -m src.gui
```

If Ollama is installed but its server is not running, run `ollama serve` in a separate Terminal
window and leave it open. Skip this if the Ollama app is already running.

#### Using the app (Windows and macOS)

Put `.txt`, `.md`, `.pdf`, or `.docx` files in `data/documents/custom`, then build the index and ask a
question:

```sh
uv run rag ingest --kb custom
uv run rag ask "What is in my documents?" --kb custom
```

Ingestion reports document-loading and embedding progress with counts, percentages,
elapsed time and estimated time remaining, followed by index-saving status.
The GUI's **Rebuild index** action also displays progress.

Embedding settings under `[models]`:

```toml
embedding_device = "auto" # CUDA when available, otherwise MPS, otherwise CPU
embedding_batch_size = 64
embedding_cache = true
embedding_cache_version = "1"
```

Use `cuda`, `mps`, or `cpu` to force a device; an unavailable forced device reports
an error. Availability depends on the installed PyTorch build and hardware.
Lower the batch size if GPU memory is insufficient. The selected device is shown
during ingestion. Both document and query embeddings use these settings.

On Windows, `uv sync` installs CUDA 12.8-enabled PyTorch from the official PyTorch
wheel index configured in `pyproject.toml`. This choice is recorded in `uv.lock`.
Other platforms use PyPI's PyTorch build, including MPS support on compatible Macs.
The CUDA build can still run on CPU when CUDA is unavailable. Restart the GUI after
updating PyTorch so it loads the newly installed libraries.

The per-knowledge-base `embeddings.sqlite3` cache reuses unchanged chunk text and
deduplicates repeated text. Each group of up to 256 new chunks is committed as a
checkpoint: rerun the same ingest command after interruption to reuse completed
work. Document scanning/chunking still runs, but cached chunks skip model encoding.
Changing the model, its resolved revision, or `embedding_cache_version` selects
a fresh cache namespace. Increase the version after changing local model weights.
Set `embedding_cache = false` to bypass caching. Existing indexes made before this
feature do not populate the cache; the first cached ingest must encode them once.
Index JSON is written incrementally to avoid a second large JSON string in memory.

To launch the browser interface:

```sh
uv run python -m src.gui
```

Open the URL printed in the terminal (normally `http://127.0.0.1:7860`). If the port is occupied,
the GUI automatically tries successive ports. Use `--port 8000` to start searching at another port. If no index exists, asking a question builds it automatically.
After adding documents, click **Rebuild index** to include them. The first ingest downloads the configured embedding model from
Hugging Face and caches it locally.

GUI follow-up questions use recent messages from the current chat, including earlier
answers. The prompt keeps up to 12 messages (8,000 characters total, at most 2,000
per message); retrieval also includes the last two retained user questions to help
resolve references. **New chat** or switching knowledge bases clears this context.
History stays in the browser session and is not saved across restarts. Separate
CLI `rag ask` commands remain independent.

### Configuration

### Choosing a knowledge base

Select **custom**, **multidoc2dial**, **anilist**, or **vndb** from the GUI's collapsed
**Advanced** menu, which also contains **Rebuild index**. Switching clears the chat
and pending uploads. Chat, uploads, and rebuilding use the selected knowledge base;
each has its own documents and index. GUI changes are saved in the settings file,
so future launches open the most recently selected knowledge base.
You can also save a selection through the CLI:

```sh
uv run rag select-kb vndb
uv run python -m src.gui
```

`select-kb` saves `[knowledge_base] default` in the settings file without rebuilding
the index. A successful `rag ingest --kb NAME` also saves that selection. Restart a
running GUI after changing it. Use `rag --config path/to/settings.toml select-kb NAME`
for another settings file. `ask --kb NAME` is a one-command override and does not
change the saved selection.

```sh
uv run rag prepare-multidoc2dial
uv run rag ingest --kb multidoc2dial
uv run rag ask "How do VA education benefits work?" --kb multidoc2dial

uv run rag prepare-anilist
uv run rag ingest --kb anilist
uv run rag ask "What is Cowboy Bebop about?" --kb anilist
```

`prepare-multidoc2dial` uses existing prepared documents when available; otherwise it
downloads the official dataset ZIP. Use `--source path/to/multidoc2dial_doc.json`
to import a local copy.

AniList requires only `anilist_anime_data_complete.csv` from the
[AniList Kaggle dataset](https://www.kaggle.com/datasets/calebmwelsh/anilist-anime-dataset).
`prepare-anilist` reuses that file in `data-raw/` or `data/raw/` when available;
otherwise it requests only that CSV from Kaggle. The CSV rows are the knowledge base
source: preparation converts them into one Markdown document per anime, including
titles, descriptions, and metadata, which `ingest --kb anilist` indexes.
No pickle or dataset scripts are needed. If Kaggle requires authentication,
download only the CSV in your browser and run:

```sh
uv run rag prepare-anilist --source path/to/anilist_anime_data_complete.csv
```

Preparation does not build embeddings. After importing updated data, rebuild the selected index.

### AniList catalog tools

AniList preparation also writes `data/documents/anilist/catalog.jsonl`. On first
use, it is imported into an indexed SQLite database beside the export. Subsequent
requests query SQLite without loading the whole catalog. The chat automatically
uses this catalog for popularity rankings, genre/tag recommendations,
recommendations based on another anime, character descriptions, and title/synopsis
lookups. Explicit `rag tool` commands execute directly without a routing or answer
model. Chat uses the semantic routing process described below.
Examples:

```sh
uv run rag ask "What are the top 10 most popular anime?" --kb anilist
uv run rag ask "Recommend horror anime with the Time Loop tag" --kb anilist
uv run rag ask "Recommend anime like Higurashi" --kb anilist
uv run rag ask "Who is Naruto?" --kb anilist
uv run rag ask "What is ひぐらしのなく頃に?" --kb anilist
```

Explicit tools are also available for scripts and debugging:

```sh
uv run rag tool popular --limit 10
uv run rag tool popular "Sci-Fi 2020" --limit 5
uv run rag tool recommend "Horror Time Loop"
uv run rag tool recommend "similar to Naruto"
uv run rag tool character "Naruto Uzumaki"
uv run rag tool title "Higurashi"
```

Popularity is sorted numerically across the imported snapshot; it is not a live
AniList ranking. Genre/tag filters require all named terms and can include a year
or format (`TV`, `movie`, `OVA`, `ONA`). Similar-anime recommendations prioritize
positive imported recommendation votes, then shared genres/tags, with popularity
as a tie breaker. Named seed anime are excluded. Every result cites its document.
Titles match English, romanized Japanese, native Japanese, and synonyms, including
partial names and normalized punctuation. Short character names are disambiguated
using the popularity of their associated anime. Missing descriptions are reported
explicitly; marked spoiler blocks are omitted from character biographies.
Tag labels remain available for intentional recommendation filters, including Tragedy.
Full character names also match reordered name components, such as `Natsuki
Subaru` and `Subaru Natsuki`. All supplied components must match before popularity
is considered. A routed `anime_scope` constrains character lookup to that franchise;
failure within that scope never widens the search to unrelated anime. Corrections
such as `I mean from Re:Zero` retain the person from the prior user request and
replace the anime scope. Earlier assistant answers do not authorize character
identities, because those answers may themselves be incorrect.

AniList chat normally uses two LLM calls per question (`src/entity_agent.py`): one to plan
all retrieval functions, and one to write the final answer. Function descriptions
are retrieved semantically, and the LLM chooses the complete plan and arguments.
The application executes the plan locally without additional LLM calls between
steps. Dependencies reference earlier results, for example `$0.matches.0.id`
or `$1.main_characters.*.id`. Database lookups and embedding operations do not
count as LLM calls. The older routers remain only for backend regression checks.

Separate anime, character, genre and tag dictionaries contain canonical IDs,
names and aliases. Indexed exact matching and fuzzy candidate search handle
punctuation, Japanese titles, reversed character names and typos. Dictionary
matches establish identity; factual answers require loaded source evidence.
Preference retrieval embeds imported tag descriptions and categories alongside
their labels. The router receives semantic feature candidates for requests such
as sad anime, rather than treating the preference phrase as a title. Filtered
candidates can then be semantically ranked against the requested preference.

`get_anime` returns a coherent bundle with scalar facts, synopsis, genres, tags,
main-character IDs and recommendation links. Follow-up questions reuse the
resolved entity and reload its bundle from SQLite. `get_character` loads a
character's description and appearances. Full cast lists, biographies, staff,
voice actors and relationships are loaded separately or paged as needed, rather
than putting every linked character biography into the initial anime bundle.

The LLM can compose these operations. For example, a request for anime with a
main character like Re:Zero can load Re:Zero, select its main character, read
that profile, and run `recommend_by_character`. That tool retrieves character
descriptions, restricts candidates to main characters, and semantically ranks
them; the LLM explains the supported similarities. Ordinary recommendations
use anime seeds or dictionary-resolved genre/tag IDs. Popularity uses numerical
catalog sorting. `search_passages` retrieves additional evidence when bundles
are insufficient, using lexical search and then vector search when needed.

Conversation state retains trusted tool-returned IDs, scoped to the exact chat
history. A new chat has no inherited subject. Recommendations preserve the
reference anime rather than replacing it with the first result. The router explicitly
sets `use_previous_context`: true to use earlier topics for a follow-up, false to
remove earlier history, evidence and entity IDs before executing an independent
request. This context decision does not add an LLM call. Invalid calls
produce structured missing-evidence information for the finalizer; conversational
answers and clarification questions are written by the LLM. There are no canned
clarification responses or planning retry loops. Repeated functions are deduplicated,
failed dependencies are skipped, and independent steps can still finish. Invalid
plans still proceed to the final LLM call, which uses available evidence or asks
its own clarification. Factual answers cite loaded source filenames. Browser history can
recover an unambiguous user-named anime after the in-memory topic cache is lost.
Complete recommendation requests without backward references start with an empty
history and entity pool before planning. Requests such as "sad anime like that"
retain conversation context. Debug `history_policy` and the live `history_selection`
event show this decision. A cached empty reference stays empty instead of reviving
an older title from browser history. This boundary adds no model calls.

`[retrieval] agent_max_steps` bounds local functions in the plan (default 8), `context_chars`
bounds retained evidence (default 14,000), and `[generation] context_window`
sets Ollama's context window (default 8,192 tokens). Older bundles outside the
evidence window are represented by reload instructions. Dictionary and guide
indexes are created automatically and reused until their source changes. Simple
facts avoid the large document vector index; chained questions run several local
functions but still use two LLM calls. Per-query `last_route` diagnostics include
`llm_calls`, `planned_tools` and `executed_tools`. `detail_top_k`, `top_k` and `router_top_k` continue
to configure the legacy retrieval paths.
If the final reply makes an unsupported alias or cast attribution, identity
validation permits one LLM correction; diagnostics report that extra call.

In the GUI, enable **Advanced → Show router debug** to inspect the latest query's
router plan, context choice, selected entities, tool arguments/results status and
LLM call count. The panel is hidden by default. Toggling it does not run another
query; diagnostics are saved in that browser session and cleared by New chat or
a knowledge-base change.
GUI request diagnostics also print automatically to the launching terminal (stderr),
including stage starts, the validated plan as soon as it is ready, each completed
tool and failures as they happen, regardless of panel visibility. Output is flushed
immediately. The terminal ends with a brief request summary; the GUI retains the full snapshot.
Debug snapshots include total request `elapsed_seconds`. Entity-agent requests also
include `timings_seconds` for context preparation, guide retrieval, feature lookup,
planning, tools and answer generation (including any identity correction). Each
tool trace entry reports its own `elapsed_seconds`, including failed or cached steps.
The conversation panel uses 75% of the viewport height (at least 600px) on desktop
and 65% on mobile to show more of the chat at once.

For CLI diagnostics, run `uv run rag ask "Recommend sad anime" --kb anilist --debug`.
The router plan, context choice, executed tools, LLM call count and elapsed time
are printed as JSON to stderr, including available diagnostics when a query fails.
The answer remains on stdout; omit `--debug` to disable these logs.

SQLite stores titles/aliases, scalar facts, genres/tags, characters, appearances,
and recommendation relationships with stable IDs. Its FTS5 passages and new vector
chunks retain `anime_id`, `entity_type`, and `character_id`. Database refreshes are
transactional and triggered by an export change or schema upgrade. The first
conversion can take tens of seconds for a large catalog; it is reused on restart.
`catalog.jsonl` remains the portable import snapshot.

Existing AniList exports automatically upgrade from a cached CSV in `data-raw/`
or `data/raw/` when the catalog is absent. Otherwise, run `prepare-anilist --source`
again with your CSV. Rebuild the vector index once to use the cleaner semantic
chunks: each retains the anime's title/aliases and character identity, while image
URLs, voice actor JSON and other export noise are excluded. Catalog answers work
immediately after preparation, even before rebuilding embeddings. Existing vector
indexes remain readable; rebuild to persist the new entity metadata. This staged
catalog path applies to AniList; other knowledge bases retain their vector flow.

### VNDB knowledge base

`prepare-vndb` reuses an extracted dump in `data-raw/vndb` or `data/raw/vndb`
when available. Otherwise it downloads `vndb-db-latest.tar.zst` from the
[official VNDB downloads](https://dl.vndb.org/dump/) into `data/raw/`, unpacks it,
and prepares the knowledge base. Subsequent runs reuse the cached archive.
The dump contains PostgreSQL COPY text tables and matching `.header` files;
Python 3.14 handles Zstandard decompression without extra dependencies or PostgreSQL.

```sh
uv run rag prepare-vndb
# Or select another extracted dump:
uv run rag prepare-vndb --source path/to/vndb-dump
# Or download a snapshot from the official site and prepare it directly:
uv run rag prepare-vndb --source path/to/vndb-db-YYYY-MM-DD.tar.zst
uv run rag ingest --kb vndb
uv run rag ask "Who developed Kana -Little Sister-?" --kb vndb
```

Archive contents are extracted into a temporary directory with safe path filtering,
then removed after preparation; the downloaded archive is retained. The manifest
records the archive path and dump timestamp. To update an existing local dump, pass
a newer archive with `--source`, then rebuild with `ingest --kb vndb`.

`src/vndb.py` joins novels with multilingual titles, aliases, release dates,
platforms, developers/publishers, staff credits, and related works. It writes one
UTF-8 Markdown document per visual novel under `data/documents/vndb`, plus a JSON
manifest recording the snapshot and exclusions. Source URLs and the dump timestamp
are included in every document. COPY escapes and nulls are decoded, BBCode links
are preserved as readable text, and explicitly marked synopsis spoilers are omitted.
Language and platform identifiers retain the dump's codes; relation codes are
expanded into readable labels.

The enriched export includes tag definitions, aliases, category and parent labels
(including setting/theme descriptors), linked anime metadata, release notes and
engine descriptions. Tags are accepted when their nonignored, non-lie raw votes
have a positive mean; the export records counts and the raw mean, which is **not**
VNDB's official weighted relevance score. User identities and vote notes are omitted.
Anime metadata is from AniDB and carries the dump's CC-BY-NC-SA terms.

Characters get separate `character-c<number>-v<number>.md` documents for each linked
VN, with names, aliases, description, roles, release IDs, available basic profile
fields, traits and voice cast. Shared tag/trait definitions are separate Markdown
documents, keeping repeated descriptions out of every VN and character record.
The VN document contains the corresponding character
references. Spoiler-flagged character appearances, aliases, tags and traits above
the chosen threshold are excluded. The default is `--spoiler-level 0`; use `1` for
minor spoilers or `2` for major spoilers. Explicit BBCode spoiler blocks remain
omitted at every threshold. Unmarked descriptions can still contain spoilers;
this export does not enforce chapter/route-specific reader progress.

The dump contains **no reviews**, and Kana's documented API has no review endpoint.
To include reviews you already have permission to use, supply UTF-8 JSONL (one
JSON object per line):

```json
{"id":"example-1","vn_id":"v1","author":"Reader","date":"2026-10-14","source_url":"https://example.org/reviews/1","spoiler_level":0,"text":"Review text goes here."}
```

```sh
uv run rag prepare-vndb --reviews data-raw/reviews.jsonl --spoiler-level 0
```

All shown fields are required. Review IDs must be unique and contain only letters,
digits, underscores or hyphens; VN IDs must exist in the dump. Reviews are separate
documents labelled as reader opinions, with source/author/date and VN references.
No reviews are downloaded automatically. Omit `--reviews` on a later preparation
and previously generated review documents are removed.

`catalog.jsonl` contains one structured record per VN with tags, linked characters,
anime, review provenance, ratings/play times and release languages/platforms. It is
not embedded as prose. It provides data for future recommendation filtering;
the current RAG still uses vector similarity and does not yet apply these filters.

Re-running preparation updates documents and removes converter-owned VN,
character and review files absent from the new export. Other files are preserved. Rebuild the index
after preparing a new snapshot. The dump's `README.txt` documents ODbL/DbCL licensing
and separate terms that may apply to descriptions; keep the original license files.

Set `[knowledge_base] default` in `settings.toml` to choose the default (currently `multidoc2dial`).
Custom documents now live in `data/documents/custom`; move any personal files previously placed
directly in `data/documents` into that folder and rebuild the custom index.

### Other settings

Edit `settings.toml` to change document and index paths, prompt, models, chunking, retrieval, and
generation settings. The environment variables `RAG_LLM_MODEL`, `RAG_EMBEDDING_MODEL`, and
`RAG_OLLAMA_URL` temporarily override model settings.

Use another TOML file with `--config path/to/settings.toml`. The provider boundary is
`src/providers.py`, so another local generation server can be integrated without
changing document loading, indexing, prompting, or the CLI.
