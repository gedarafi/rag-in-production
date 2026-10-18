"""Small, catalog-grounded semantic decisions for preference and attribute clauses."""
import json
import re

import numpy as np

from .anime_tools import normalize, contains
from .request_constraints import catalog_text_values, mentioned_features, validate_request
from .request_interpreter import DECOMPOSITION_SCHEMA, FIELD_MEANINGS, decomposition_prompt
from .sqlite_catalog import STOP
from .semantic_attributes import interpret_attributes, covered_property, retrieve_attributes


ROLE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['role'],
               'properties': {'role': {'enum': ['required', 'excluded', 'preferred', 'irrelevant']}}}

# Attribute vocabulary belongs to the schema, not to individual user requests.
ATTRIBUTE_TERMS = {
    'start_date': ('year', 'date', 'release', 'released', 'air', 'aired', 'premiere', 'premiered'),
    'year': ('year', 'years', 'date', 'release', 'released', 'air', 'aired', 'premiere', 'premiered'),
    'episodes': ('episode', 'episodes'),
    'duration': ('minute', 'minutes', 'hour', 'hours', 'runtime', 'duration', 'length', 'long'),
    'score': ('score', 'rating', 'rated'), 'popularity': ('popularity', 'popular', 'viewers'),
}
VALUE_ALIASES = {
    ('format', 'MOVIE'): ('film', 'films', 'movie', 'movies'),
    ('format', 'TV'): ('tv', 'television'),
    ('status', 'FINISHED'): ('finished', 'completed', 'ended'),
    ('status', 'RELEASING'): ('airing', 'ongoing'),
}
PREFERENCE_STOP = STOP | {'story', 'stories', 'theme', 'themes', 'setting', 'settings', 'content',
                          'anime', 'animation', 'movie', 'movies', 'film', 'films'}


def role_prompt(question, concept):
    return '''Classify the role of ONE concept in the user request. Read negation and meaning.
Examples: "Find games without combat", concept=combat -> excluded.
"Find history books", concept=history -> required.
"I'd prefer a gentle book", concept=gentle -> preferred.
"Stories that weren't set in space", concept=space -> excluded.
"Find funny films", concept=funny -> preferred.
Use irrelevant if the concept is not requested. Return only JSON.
REQUEST: ''' + question + '\nCONCEPT: ' + concept


def attribute_filters(call, question, fields, text_values, details=None, retrieved=None, embed=None):
    """Interpret measurable attributes separately from thematic preferences."""
    filters, preferences, notes, sources = interpret_attributes(call, question, fields, FIELD_MEANINGS, retrieved or {}, embed)
    if details is not None:
        details.update(numeric_preferences=preferences, limitations=notes, sources=sources)
    source = normalize(question)
    # These are schema-defined categorical attributes, not theme keywords.
    for field in ('format', 'source_material', 'status'):
        values = text_values.get(field)
        if not values:
            continue
        values = [value for value in values if any(contains(source, normalize(term)) for term in
                  (value, value + 's', *VALUE_ALIASES.get((field, value), ())))]
        if not values:
            continue
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['applies', 'operator', 'value'],
                  'properties': {'applies': {'type': 'boolean'}, 'operator': {'enum': ['eq', 'ne']},
                                 'value': {'enum': values}}}
        answer = call('Determine whether this ONE catalog attribute was explicitly requested. Do not infer a '
                      'format, source or status from genre or mood. If not requested use applies=false. '
                      'Use only existing values. eq requires it; ne excludes it. Return JSON.\n' + json.dumps(
                          {'field': field, 'meaning': FIELD_MEANINGS[field], 'values': values, 'request': question}), schema)
        if answer.get('applies'):
            filters.append({'field': field, 'operator': answer['operator'], 'value': answer['value'],
                            'source_text': question, 'mapping': 'exact'})
    return filters


def lexical_preference(agent, phrase, features):
    """Use existing feature examples as vocabulary, with the requested hierarchy."""
    path = agent.settings.root / 'prompts/features.json'
    examples = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
    tokens = set(normalize(phrase).split()) - PREFERENCE_STOP
    if not tokens:
        return None
    literal = [dict(field='genres' if kind == 'genre' else 'tags', value=name, meaning=name,
                    coverage=len(normalize(name).split()) / len(tokens))
               for kind, _, name in features if contains(normalize(phrase), normalize(name))]
    if literal:
        return sorted(literal, key=lambda row: (-len(normalize(row['value'])), row['field'] != 'genres'))[0]
    matches = []
    for kind, identifier, name in features:
        texts = [name] + [value for row in examples if row['kind'] == kind and row['id'] == identifier
                         for value in row['examples']]
        coverage = max((len(tokens & set(normalize(text).split())) / len(tokens) for text in texts), default=0)
        if coverage == 1:
            matches.append({'field': 'genres' if kind == 'genre' else 'tags', 'value': name,
                            'meaning': name, 'coverage': coverage})
    return next((m for m in matches if m['field'] == 'genres'), next(iter(matches), None))


def interpret(agent, question, history):
    resolution = None
    def call(prompt, schema):
        if resolution is not None:
            prompt += ('\nConfirmed mention roles (reference data): ' + json.dumps(resolution, ensure_ascii=False) +
                       '\nNamed reference titles/characters are not content filters merely because their names also match categories.')
        agent.last_usage['llm_calls'] += 1
        raw = agent.models.interpret_request(prompt, schema)
        agent.last_usage.setdefault('request_interpretation_outputs', []).append(raw)
        return json.loads(raw)

    typed_mentions = None
    if callable(getattr(agent.models, 'classify_mentions', None)):
        from .mention_interpreter import classify
        decomposition, typed_mentions = classify(agent, question, history)
    else:
        decomposition = call(decomposition_prompt(question, history), DECOMPOSITION_SCHEMA)
    if not isinstance(decomposition, dict) or set(decomposition) != set(DECOMPOSITION_SCHEMA['required']):
        raise ValueError('Invalid request decomposition')
    if decomposition['intent'] not in ('recommendation', 'factual', 'other') or type(decomposition['use_previous_context']) is not bool:
        raise ValueError('Invalid request decomposition intent')
    for key in ('required', 'excluded', 'preferred', 'entity_queries'):
        if not isinstance(decomposition[key], list) or len(decomposition[key]) > 8 or any(not isinstance(v, str) for v in decomposition[key]):
            raise ValueError('Invalid request clause list')
    from .entity_resolution import resolve_mentions
    resolution = resolve_mentions(agent, question, history, decomposition, typed_mentions=typed_mentions)
    if resolution is not None:
        decomposition['entity_queries'] = [s['name'] for s in resolution['subjects']]
        # Classification can correct either direction: a title initially mistaken
        # for a category, or a category initially mistaken for a title.
        for role in ('required', 'excluded', 'preferred'):
            decomposition[role] = []
        for mention in resolution['mentions']:
            if mention['role'] in ('required', 'excluded', 'preferred'):
                target = decomposition[mention['role']]
                if mention['text'] not in target:
                    target.append(mention['text'])
    base = {'intent': decomposition['intent'], 'use_previous_context': decomposition['use_previous_context'] and bool(history),
            'entity_queries': decomposition['entity_queries'], 'filters': [], 'soft_query': '', 'limitations': []}
    if base['intent'] != 'recommendation':
        return {**base, 'mappings': [], **({'entity_resolution': resolution} if resolution is not None else {})}
    fields = agent.catalog.searchable_fields()
    features = agent.catalog.query("SELECT kind,id,name FROM entity_dictionary WHERE kind IN ('genre','tag') ORDER BY kind,id")
    if not hasattr(agent, '_constraint_text_values'):
        agent._constraint_text_values = catalog_text_values(agent.catalog, fields)
    attributes = {}
    phrases = [phrase for key in ('entity_queries', 'required', 'excluded', 'preferred') for phrase in decomposition[key]
               if normalize(phrase) not in PREFERENCE_STOP and normalize(phrase) in normalize(question)]
    retrieved = retrieve_attributes(agent.embed, question, phrases, fields, features)
    agent.last_usage['attribute_retrieval'] = retrieved
    base['filters'] = attribute_filters(call, question, fields, agent._constraint_text_values, attributes, retrieved, agent.embed)
    base['limitations'].extend(attributes['limitations'])
    property_sources = attributes['sources']
    base['entity_queries'] = [phrase for phrase in base['entity_queries']
                              if not covered_property(phrase, property_sources, PREFERENCE_STOP)]
    feature_question = question
    if resolution is not None:
        # Remove only confirmed title/name mentions from automatic feature discovery.
        # Preserve remaining wording, including genuine category preferences.
        for subject in resolution['subjects']:
            feature_question = feature_question.replace(subject['text'], ' ', 1)
    mentions = mentioned_features(feature_question, features)
    source = normalize(question)
    phrases = {}
    for phrase in decomposition['required'] + decomposition['excluded'] + decomposition['preferred']:
        if contains(source, normalize(phrase)):
            phrases.setdefault(normalize(phrase), phrase)
    for _, _, name in mentions:
        phrases.setdefault(normalize(name), name)
    if len(phrases) > 16:
        raise ValueError('Too many independent request clauses')
    preferred = {normalize(p) for p in decomposition['preferred']}
    domain_types = {normalize(value) for value in agent._constraint_text_values.get('type', [])}
    metadata = []
    for field, kind in fields.items():
        if kind == 'date':
            continue  # Explicit full dates are already handled; preserve scalar/theme retrieval.
        if field in ('genres', 'tags') or re.search(r'title|description|image|url|thumbnail|synonym|alias', field, re.I):
            continue
        values = agent._constraint_text_values.get(field)
        if kind == 'text' and not values:
            continue
        meaning = FIELD_MEANINGS.get(field, field.replace('_', ' ')) + (' ' + ' '.join(values) if values else '')
        metadata.append({'field': field, 'kind': kind, 'meaning': meaning, 'values': values})
    if not hasattr(agent, '_constraint_field_vectors'):
        agent._constraint_field_vectors = agent.embed([row['meaning'] for row in metadata])
    soft, description_exclusions, hierarchy = [], [], []
    covered_terms = set()
    for constraint in base['filters']:
        covered_terms.update(ATTRIBUTE_TERMS.get(constraint['field'], (constraint['field'],)))
    for phrase in phrases.values():
        if covered_property(phrase, property_sources, PREFERENCE_STOP):
            continue
        if normalize(phrase) in domain_types:
            continue
        if re.search(r'\d', phrase) and any(fields[f['field']] in ('number', 'date') for f in base['filters']):
            continue
        if any(f['field'] in ('format', 'source_material', 'status') and
               ((set(normalize(phrase).split()) - PREFERENCE_STOP) <= set(normalize(f['value']).split()) or
                normalize(phrase) == normalize(f['value']) + 's')
               for f in base['filters']):
            continue
        if not (set(normalize(phrase).split()) - PREFERENCE_STOP):
            continue
        if (set(normalize(phrase).split()) - PREFERENCE_STOP) <= covered_terms:
            continue
        exact = [dict(field='genres' if kind == 'genre' else 'tags', value=name, meaning=name)
                 for kind, _, name in features if normalize(name) == normalize(phrase)]
        if exact:
            # Exact catalog membership is structural. Only request polarity
            # needs an LLM decision; it cannot substitute a neighboring feature.
            candidate = sorted(exact, key=lambda r: r['field'] != 'genres')[0]
            role = call(role_prompt(question, phrase), ROLE_SCHEMA)['role']
            expanded = any(contains(normalize(p), normalize(phrase)) and normalize(p) != normalize(phrase)
                           for key in ('required', 'excluded', 'preferred', 'entity_queries') for p in decomposition[key])
            decision = {'candidate': candidate, 'role': role, 'mapping': 'broader' if expanded else 'exact'}
        elif normalize(phrase) in preferred and not any(
                row.get('similarity', 0) >= .8 for kind in ('genre', 'tag')
                for row in agent.feature_candidates(phrase, kind=kind, limit=1)):
            # Subjective effects are semantic ranking inputs, never literal SQL.
            # Retrieve genres first, then tags, and keep descriptions as fallback.
            decision = {'candidate': lexical_preference(agent, phrase, features),
                        'role': 'preferred', 'mapping': 'broader'}
        elif lexical_preference(agent, phrase, features):
            decision = {'candidate': lexical_preference(agent, phrase, features),
                        'role': call(role_prompt(question, phrase), ROLE_SCHEMA)['role'], 'mapping': 'broader'}
        else:
            options = []
            for kind in ('genre', 'tag'):
                for row in agent.feature_candidates(phrase, kind=kind, limit=3):
                    options.append({'field': 'genres' if kind == 'genre' else 'tags',
                                    'value': row['name'], 'meaning': row.get('description', row['name']),
                                    'confidence': row.get('similarity', 0)})
            scores = agent.embed([phrase])[0] @ agent._constraint_field_vectors.T
            options += [metadata[i] for i in np.argsort(scores)[::-1][:3]]
            options.append({'field': 'description', 'meaning': 'Semantic description matching; no exact category'})
            choice_schema = {'type': 'object', 'additionalProperties': False,
                             'required': ['choice', 'role', 'operator', 'value', 'mapping'],
                             'properties': {'choice': {'enum': list(range(len(options)))},
                                            'role': ROLE_SCHEMA['properties']['role'],
                                            'operator': {'enum': ['eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'contains', 'not_contains']},
                                            'value': {'type': 'string'}, 'mapping': {'enum': ['exact', 'broader']}}}
            prompt = ('Resolve ONE property using existing catalog choices. Preference hierarchy: Genre, then Tags, '
                      'then General Descriptions. Choose a genre only if it actually expresses the concept; otherwise '
                      'try tags, then semantic descriptions. Factual attributes such as release dates, counts, lengths '
                      'and formats use the corresponding structured field, independent of that preference hierarchy. '
                      'Choose one numbered option. Respect negation in the whole request. For features copy the '
                      'option value exactly; required uses eq, excluded uses ne. For scalar attributes return the '
                      'actual typed comparison and value; apply negation to the comparison. General descriptions '
                      'and subjective effects use preferred, not literal keyword filters. Return JSON.\n' +
                      json.dumps({'request': question, 'property': phrase, 'choices': options}, ensure_ascii=False))
            mapped = call(prompt, choice_schema)
            index = mapped['choice']
            if type(index) is not int or not 0 <= index < len(options):
                raise ValueError('Invalid catalog choice')
            candidate = options[index]
            # A real category name is not enough: its mapping must also be
            # supported by the request. Weak neighbors remain description hints.
            if candidate['field'] in ('genres', 'tags') and candidate.get('confidence', 0) < .65:
                candidate = {'field': 'description', 'meaning': 'Unverified semantic description match'}
            decision = {'candidate': candidate, 'role': mapped['role'], 'mapping': mapped['mapping']}
            if candidate['field'] == 'description':
                original_role = call(role_prompt(question, phrase), ROLE_SCHEMA)['role']
                decision['candidate'] = None
                decision['role'] = 'preferred'
                if original_role == 'excluded':
                    description_exclusions.append(phrase)
                    base['limitations'].append('Description matching can downrank ' + repr(phrase) +
                                              ', but the catalog cannot verify its complete absence.')
                    decision['role'] = 'irrelevant'
                elif original_role == 'required':
                    base['limitations'].append('The exact requirement ' + repr(phrase) +
                                              ' is assessed semantically from descriptions, not verified by a catalog category.')
            else:
                decision['role'] = call(role_prompt(question, phrase), ROLE_SCHEMA)['role']
                decision.update(operator=mapped['operator'], value=candidate.get('value', mapped['value']))
        role = decision['role']
        if role not in ('required', 'excluded', 'preferred', 'irrelevant'):
            raise ValueError('Invalid requirement role')
        hierarchy.append({'phrase': phrase, **decision})
        if role == 'irrelevant':
            continue
        if role == 'preferred':
            soft.append(phrase)
            if decision['candidate'] and decision['candidate'].get('value'):
                soft.append(decision['candidate']['value'])
            continue
        candidate = decision['candidate']
        if candidate is None:
            base['limitations'].append('The catalog cannot verify this hard requirement: ' + phrase + '.')
            continue
        field = candidate['field']
        op = ('ne' if role == 'excluded' else 'eq') if field in ('genres', 'tags') else decision.get('operator', 'eq')
        base['filters'].append({'field': field, 'operator': op,
                                'value': candidate.get('value', decision.get('value', phrase)),
                                'source_text': question, 'mapping': decision['mapping']})
    base['soft_query'] = ' '.join(dict.fromkeys(soft))
    result = validate_request(base, question, history, fields, features, agent._constraint_text_values)
    result['preference_hierarchy'] = hierarchy
    result['description_exclusions'] = description_exclusions
    result['numeric_preferences'] = attributes['numeric_preferences']
    if resolution is not None:
        result['entity_resolution'] = resolution
    agent.last_usage['request_decomposition'] = decomposition
    return result
