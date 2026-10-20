# Contextual entity resolution

The AniList assistant uses two LLM decisions before tool planning:

1. `mention_interpreter.py` classifies names and criteria from the full request and
   selected conversation context. It runs **before dictionary lookup**, so a catalog
   alias cannot turn an ordinary word into a title. Mentions carry a type, role,
   lookup phrase, and short contextual meaning.
2. `entity_resolution.py` retrieves candidates of that type and asks the model to
   select catalog identities. The output schema contains readable canonical names
   and eligible IDs. Code validates each selection against its own phrase's
   candidates. Lookup cannot change the previously classified role.

Types include anime, character, genre, tag, description, and other metadata.
Roles distinguish factual subjects, recommendation references, positive criteria,
exclusions, preferences, and ambiguity. Unmapped descriptions remain descriptions.
An ambiguous named entity produces the model's clarification question, rather than
selecting the first candidate. No new word blacklist is used. For example, Air and
Original remain valid titles when the user actually means those works.

Anime candidate retrieval includes longer title/alias prefixes as well as exact
aliases, with year, format, and synopsis context. This matters when a short alias
also belongs to an unrelated entry: Demon Slayer is an alias of Onigiri in the
imported snapshot, but it also prefixes the familiar Demon Slayer TV titles.
Candidate retrieval does not decide which identity the request means.

Only confirmed anime/character subjects seed the active scope. Loading cast,
recommendations, relations, or another tool target adds evidence and known IDs;
it does not add mandatory answer subjects. Factual grounding restores confirmed
subjects before generation and supports mixed anime/character comparisons without
giving incidental characters priority over anime. Referential followups can
rehydrate trusted conversation state. The old direct-matching path remains only
for legacy adapters without the contextual LLM methods.

Both factual and recommendation requests share this path. Existing typed catalog
filter validation remains responsible for attribute types, negation, numerical
bounds, and exact versus approximate category mappings. Subsequent preference and
planning prompts receive the confirmed mention roles.

## Verification and diagnostics

```sh
uv run pytest tests/recommender/test_entity_resolution.py
uv run python scripts/check_entity_resolution.py
```

The live script uses the configured local model and saves decisions under
`data/evaluation/hard_facts/diagnostics/entity_resolution_live.json`. Debug usage
includes `typed_mentions`, `entity_resolution`, raw resolver responses, and any
validation-guided retry. The model sees catalog candidates, not benchmark answers.

These stages add model work and are still model-dependent. Run a fresh benchmark
under a new output name after changing the pipeline: existing checkpoints are
historical results and are not automatically retried. A process already running
the old implementation must be restarted to load these changes.
