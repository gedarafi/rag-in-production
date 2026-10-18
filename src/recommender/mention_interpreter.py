"""Contextual mention classification before exposing any catalog matches."""
import json

from .anime_tools import normalize, contains

KINDS = ['anime', 'character', 'genre', 'tag', 'description', 'other']
ROLES = ['subject', 'reference', 'required', 'excluded', 'preferred', 'ambiguous']
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'required': ['intent', 'use_previous_context', 'mentions'], 'properties': {
              'intent': {'enum': ['recommendation', 'factual', 'other']},
              'use_previous_context': {'type': 'boolean'},
              'mentions': {'type': 'array', 'maxItems': 16, 'items': {
                  'type': 'object', 'additionalProperties': False,
                  'required': ['meaning', 'text', 'kind', 'role', 'query'],
                  'properties': {'text': {'type': 'string'}, 'query': {'type': 'string'},
                                 'kind': {'enum': KINDS}, 'role': {'enum': ROLES},
                                 'meaning': {'type': 'string', 'maxLength': 180}}}}}}


def examples():
    def example(text, intent, mentions):
        return {'request': text, 'output': {'intent': intent, 'use_previous_context': False,
                                          'mentions': [dict(text=t, query=t, kind=k, role=r, meaning=m)
                                                       for t, k, r, m in mentions]}}
    return [
        example('What year did the original Naruto anime begin?', 'factual', [('Naruto', 'anime', 'subject', 'Original TV adaptation')]),
        example('What is Air about?', 'factual', [('Air', 'anime', 'subject', 'Requested title')]),
        example('When did Cowboy Bebop first air?', 'factual', [('Cowboy Bebop', 'anime', 'subject', 'TV series; air is a verb')]),
        example('Recommend horror anime.', 'recommendation', [('horror', 'genre', 'required', 'Desired genre')]),
        example('Recommend anime without Time Loop.', 'recommendation', [('Time Loop', 'tag', 'excluded', 'Excluded theme')]),
        example('Something comforting like Monster', 'recommendation', [('comforting', 'description', 'preferred', 'Mood'), ('Monster', 'anime', 'reference', 'Reference title')]),
    ]


def instructions():
    return (
        'Extract the named works/people and content criteria from the LAST request. '
        'Copy the name into text; query is the clean name for database search, retaining sequel/season/year '
        'qualifiers that distinguish the requested work. A first/original season uses the base series name. '
        'Do not extract question words, verbs, generic anime/show wording, or version qualifiers as entities. '
        'Keep adaptation/season/year qualifications in meaning. No blacklist: a common word can be a genuine title. '
        'For factual questions use role subject. Recommendations use reference for example titles and '
        'required/excluded/preferred for content criteria. Broad categories are genre; specific themes tag; '
        'vague preferences description; numeric/format/studio/source criteria other. '
        'Use previous context only for actual followups. Assistant prose is not factual evidence. '
        'Meaning describes the request, not factual claims about the work. Return only JSON.')


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
            extraction_prompt += ('\nPrevious extraction: ' + raw + '\nValidation error: ' + str(error) +
                                  '\nCorrect the extraction. text must copy words from the current request, '
                                  'not examples or explanations. Extract actual named works/people and content criteria only.')
        else:
            discarded = [m for m in parsed['mentions'] if m not in mentions]
            if discarded:
                agent.last_usage.setdefault('unsupported_mentions', []).extend(discarded)
            agent.last_usage['typed_mentions'] = mentions
            return decomposition, mentions


def validate(result, question, history):
    if not isinstance(result, dict) or set(result) != {'intent', 'use_previous_context', 'mentions'}:
        raise ValueError('Invalid contextual mention interpretation')
    if result['intent'] not in ('recommendation', 'factual', 'other') or type(result['use_previous_context']) is not bool:
        raise ValueError('Invalid contextual mention intent')
    if not isinstance(result['mentions'], list) or len(result['mentions']) > 16:
        raise ValueError('Invalid contextual mentions')
    decomposition = {'intent': result['intent'], 'use_previous_context': result['use_previous_context'] and bool(history),
                     'entity_queries': [], 'required': [], 'excluded': [], 'preferred': []}
    source = normalize(question + ' ' + ' '.join(m['content'] for m in history
                       if m['role'] == 'user' and decomposition['use_previous_context']))
    kept = []
    discarded = []
    seen = set()
    for mention in result['mentions']:
        if not isinstance(mention, dict) or set(mention) != {'meaning', 'text', 'kind', 'role', 'query'}:
            raise ValueError('Invalid contextual mention')
        if any(not isinstance(v, str) for v in mention.values()) or mention['kind'] not in KINDS or mention['role'] not in ROLES:
            raise ValueError('Invalid contextual mention type/role')
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
