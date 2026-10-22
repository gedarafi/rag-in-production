"""Contextual mention classification before exposing any catalog matches."""
import json

from .anime_tools import normalize, contains
from .llm_schemas import MentionInterpretation, json_schema, parse
from .prompt_library import library, prompts_for

SCHEMA = json_schema(MentionInterpretation)


def examples(prompts=None):
    """Few-shot demonstrations, from classify_mentions_examples.json."""
    return (prompts or library()).data('classify_mentions_examples.json')


def instructions(prompts=None):
    return (prompts or library()).render('classify_mentions_system')


def prompt(question, history):
    return json.dumps({'history': history, 'LAST_REQUEST': question}, ensure_ascii=False)


def classify(agent, question, history):
    extraction_prompt = prompt(question, history)
    for attempt in range(2):
        agent.last_usage['llm_calls'] += 1
        raw = agent.models.classify_mentions(extraction_prompt, SCHEMA)
        agent.last_usage.setdefault('request_interpretation_outputs', []).append(raw)
        try:
            parsed = json.loads(raw)
            decomposition, mentions = validate(parsed, question, history)
        except (ValueError, TypeError) as error:
            if attempt:
                raise
            agent.last_usage.setdefault('mention_classification_corrections', []).append(str(error))
            extraction_prompt += prompts_for(agent.settings).render('classify_mentions_correction', raw=raw, error=str(error))
        else:
            discarded = [m for m in parsed['mentions'] if m not in mentions]
            if discarded:
                agent.last_usage.setdefault('unsupported_mentions', []).extend(discarded)
            agent.last_usage['typed_mentions'] = mentions
            return decomposition, mentions


def validate(result, question, history):
    parse(MentionInterpretation, result)
    decomposition = {'intent': result['intent'], 'use_previous_context': result['use_previous_context'] and bool(history),
                     'entity_queries': [], 'required': [], 'excluded': [], 'preferred': []}
    source = normalize(question + ' ' + ' '.join(m['content'] for m in history
                       if m['role'] == 'user' and decomposition['use_previous_context']))
    kept = []
    discarded = []
    seen = set()
    for mention in result['mentions']:
        if not normalize(mention['text']) or not contains(source, normalize(mention['text'])) or not mention['query'].strip():
            discarded.append(mention['text'])
            continue
        identity = (normalize(mention['text']), normalize(mention['query']), mention['kind'], mention['role'])
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(mention)
        role = mention['role']
        if mention['kind'] in ('anime', 'character') and role in ('subject', 'reference', 'ambiguous'):
            if mention['query'] not in decomposition['entity_queries']:
                decomposition['entity_queries'].append(mention['query'])
        elif role in ('required', 'excluded', 'preferred'):
            decomposition[role].append(mention['text'])
    if result['mentions'] and not kept:
        raise ValueError('Mention must be supported by the user request or selected user history: ' + repr(discarded))
    return decomposition, kept
