# Warm Clannad request latency measurements

Measured locally on 2026-10-15. Application code and settings were not changed.
Model: llama3.2:3b, Q4_K_M; context: 8192; NVIDIA GTX 1660 Ti, 6 GiB.
Ollama server log reports all 29 layers on GPU. GPU memory snapshot during inference:
5430 / 6144 MiB. Runtime model-load durations in the experiment were 5–18 ms.

## Method and limits

Ran the existing pipeline for `What was Clannad about?` with empty history,
intercepting the existing HTTP transport to save unmodified request payloads and
Ollama response metrics. The original ten message bodies are unavailable, so this
is a fresh-history control, not an exact replay of the supplied 68.55 s request.
Then replayed each captured HTTP request with compact evidence, holding model,
instruction text, schema, context size, temperature, and output limits constant.
The answer ablation reused its captured response goal; it was not an end-to-end
run of the compact planner's new plan. Each inference variant was measured once.
Prompt cache state differed and is reported below; results are illustrative,
not a statistically validated latency guarantee or a general quality evaluation.

## Model-call measurements

| Metric | Current planning | Compact planning | Current answer | Compact answer |
|---|---:|---:|---:|---:|
| User-message characters | 21379 | 9422 | 15667 | 4844 |
| System-message characters | 904 | 904 | 262 | 262 |
| Schema characters (separate API field) | 3941 | 3941 | 2 (absent) | 2 (absent) |
| Full prompt tokens | 6204 | 2417 | 4614 | 1079 |
| Reused prefix tokens, server log | 2461 | 15 | 15 | 15 |
| Newly processed input tokens, server log | 3743 | 2402 | 4599 | 1064 |
| Prompt processing seconds | 20.813 | 9.798 | 21.013 | 4.056 |
| Output tokens | 54 | 75 | 106 | 78 |
| Output generation seconds | 1.084 | 1.201 | 1.945 | 1.131 |
| Ollama total seconds | 22.302 | 11.206 | 23.213 | 5.311 |
| Client wall seconds | 24.345 | 13.242 | 25.287 | 7.364 |
| Wall minus Ollama total seconds | 2.043 | 2.036 | 2.074 | 2.054 |

Output ended normally (`stop`) for all four requests. No output-token cap was hit.
The compact variants had less cache reuse, not more, than the current planner.
API prompt_eval_count reports full prompt lengths in these responses; server log
timings distinguish the number of uncached tokens actually processed.

## Independent connection measurements

Three fresh requests per hostname to the cheap `/api/version` endpoint:

- localhost: 2.0404, 2.0497, 2.0273 seconds (mean 2.0391).
- 127.0.0.1: 0.0137, 0.0159, 0.0147 seconds (mean 0.0147).

The measured hostname-related difference is about 2.0244 seconds per HTTP request,
or 4.05 seconds per two-call reply. The underlying DNS/IPv6/connection cause was
not separately isolated. Both addresses returned Ollama version 0.40.4.

## Local answer validation

Current generate_answer stage: 34.061 s. HTTP call: 25.287 s. Remainder: 8.775 s.
Standalone unprofiled identity_issues on the captured answer/evidence: 8.504 s.
On compact evidence, checking the same answer: 0.00398 s. Both returned no issues.

cProfile identifies 50,221 finditer calls and 50,221 regex compilations. Profiling
inflates elapsed time to about 22.8 s; use the unprofiled measurement for latency.
The identity checker recursively collects anime records, including recommendation
candidates nested in an overview, then checks character-name/title combinations.
It compiles patterns for names and titles absent from the response. This creates
large CPU work even though no corrective LLM call is needed.

## Prompt attribution

Current planner user message: 4125 chars of instructions; evidence 10615;
tool descriptions 1889; repeated retrieved descriptions 964; feature candidates
1686; current-message IDs 824; known IDs 824, plus other metadata/JSON overhead.

Current answer user message: 3615 chars of instructions; evidence 10929;
primary entity facts 795, plus other metadata/JSON overhead.

Compact planning removed retrieved_guides and duplicate current_message_entity_ids,
and replaced full evidence bodies with identity and available-field summaries.
Compact answering retained identity, aliases, source, and synopsis/description;
removed nested recommendation/cast/relationship data; and deduplicated entities.
Instruction text was not shortened. Both answers remained short Clannad synopses;
broader correctness and follow-up behavior were not tested.

## Conclusions and priorities

1. Use the loopback IPv4 URL where appropriate: about 4 s saved per reply here,
   without changing model content. Configuration must still address the intended server.
2. Optimize identity validation by eliminating irrelevant regex work while retaining
   its grounding checks. About 8.5 s is avoidable in this measured synopsis case;
   do not simply disable validation or assume compact evidence is safe for all intents.
3. Give the planner compact evidence availability/identity summaries; select final
   evidence according to the question and remove duplicate bundles. Current calls
   spent 41.83 s processing input versus only 3.03 s generating output.
4. Keep history handling as a separately measured follow-up task: large costs occur
   even with zero history. The original history contribution remains unmeasured.

Current planning + answer stages totaled 58.41 s in the control. Compact HTTP calls
plus the separately measured compact validation totaled about 20.61 s. This is a
component comparison, not an end-to-end optimized application benchmark. Replacing
localhost could save another ~4.05 s under the measured connection behavior.

These changes can preserve exactly one planning call and one final-answer call.
Reducing max output tokens is a low priority for these samples; decoding is small.

Raw results: metrics.json, connections.json, stages.json, *-request.json,
*-breakdown.json, validation-unprofiled.json, validation-profile.txt,
and ollama-server-snapshot.log. Diagnostic scripts are included for reproduction
from the repository root using its virtual-environment Python.

## Implemented optimization verification

Application changes now use the IPv4 loopback URL, compact planner evidence,
deduplicate final evidence, trim nested recommendation/cast links for narrowly
recognized synopsis requests, and prefilter identity regexes using case-insensitive
literal matches. Full original evidence still goes to the identity validator.
Other question types retain recommendation reasons and cast/profile evidence.

Live optimized fresh-history Clannad run: planning 2.876 s, local tools 0.012 s,
answer stage 4.744 s: 7.63 s for planning/tools/answer. Model prompts contained
2475 and 1452 tokens. Connection overhead was 0.004 / 0.015 s per call.
This run benefited from previously populated prompt caches, so 7.63 s is a warm
sample, not a guaranteed latency or a cache-controlled before/after comparison.
New-process embedding initialization is excluded, as requested.

On the exact original captured reply and full evidence, the optimized validator
took 0.0470 s versus 8.5038 s before, returning the same empty issue list.
The live run still used exactly one plan call and one answer call.
