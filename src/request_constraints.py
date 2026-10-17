"""Typed, evidence-bearing request interpretation shared by planning and retrieval."""
import json
import re

from .anime_tools import normalize, contains
from .catalog_search import predicates, field_expression


def catalog_text_values(catalog, fields):
    """Expose actual low-cardinality scalar vocabularies to constrained decoding."""
    result = {}
    for field, kind in fields.items():
        if kind != 'text':
            continue
        expression, params = field_expression(field, fields)
        rows = catalog.query('SELECT DISTINCT ' + expression + ' FROM anime a WHERE ' +
                             expression + ' IS NOT NULL LIMIT 41', [*params, *params])
        values = [str(row[0]) for row in rows]
        if 0 < len(values) <= 40:
            result[field] = values
    return result


def mentioned_features(question, features):
    text = normalize(question)
    matches = [(row, [(match.start(), match.end()) for match in
                       re.finditer(r'\b' + re.escape(normalize(row[2])) + r'\b', text)]) for row in features]
    return [row for row, spans in matches if spans and any(
        not any(start <= left and right <= end and (start, end) != (left, right)
                for other, ranges in matches if len(normalize(other[2])) > len(normalize(row[2]))
                for start, end in ranges)
        for left, right in spans)]


def merge_planner_attributes(request, args, question, fields, text_values):
    """Preserve source-backed structured attributes independently of preferences.

    The preference resolver may not relax an existing date/length/format bound.
    Free-form prose and taxonomy guesses are deliberately not imported here.
    """
    proposed = list(args.get('filters', []))
    for key, field, op in (('year', 'year', 'eq'), ('year_from', 'year', 'gte'), ('year_to', 'year', 'lte'),
                           ('format', 'format', 'eq'), ('studio', 'studios', 'contains'),
                           ('source_material', 'source_material', 'eq')):
        if key in args:
            proposed.append({'field': field, 'operator': op, 'value': str(args[key])})
    source = normalize(question)
    for item in proposed:
        field = item.get('field')
        if field in ('genres', 'tags') or field not in fields:
            continue
        # Ignore overridden planner guesses before type validation. In particular,
        # ISO dates supplied as numeric years must not break a valid date contract.
        if any(f['field'] == field or {field, f['field']} == {'year', 'start_date'} for f in request['filters']):
            continue
        value = str(item.get('value', ''))
        term = normalize(value)
        sourced = contains(source, term) or contains(source, term + 's')
        if not sourced:
            continue
        # Literal title/synopsis searches are handled separately, never imported
        # from a planner's interpretation of a subjective preference.
        if re.search(r'title|description|synopsis|summary|name', field, re.I):
            continue
        if fields[field] == 'text' and field not in text_values:
            continue
        candidate = {'field': field, 'operator': item.get('operator'), 'value': value}
        predicates([candidate], fields)
        if not any(f['field'] == field for f in request['filters']):
            request['filters'].append(candidate)
            request['mappings'].append({**candidate, 'source_text': question, 'mapping': 'exact',
                                        'origin': 'structured_attribute_plan'})
    predicates(request['filters'], fields)
    return request


def validate_request(raw, question, history, fields, features, text_values=None):
    result = json.loads(raw) if isinstance(raw, str) else raw
    required = {'intent', 'use_previous_context', 'entity_queries', 'filters', 'soft_query', 'limitations'}
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError('Invalid request interpretation')
    if result['intent'] not in ('recommendation', 'factual', 'other') or type(result['use_previous_context']) is not bool:
        raise ValueError('Invalid request intent/context')
    for key, limit in (('entity_queries', 10), ('limitations', 8)):
        if not isinstance(result[key], list) or len(result[key]) > limit or any(not isinstance(v, str) for v in result[key]):
            raise ValueError('Invalid request ' + key)
    if not isinstance(result['soft_query'], str) or not isinstance(result['filters'], list) or len(result['filters']) > 12:
        raise ValueError('Invalid request preferences')
    # History can supply inherited constraints only when the model explicitly
    # selected it. Assistant claims never authorize a filter.
    source = normalize(question + ' ' + ' '.join(m['content'] for m in history
                       if m['role'] == 'user' and result['use_previous_context']))
    names = {}
    for kind, _, name in features:
        names.setdefault(normalize(name), []).append(('genres' if kind == 'genre' else 'tags', name))
    compiled, mappings = [], []
    filters = list(result['filters'])
    for item in filters:
        if (not isinstance(item, dict) or set(item) != {'field', 'operator', 'value', 'source_text', 'mapping'}
                or any(not isinstance(v, str) for v in item.values())
                or item['mapping'] not in ('exact', 'broader')
                or not normalize(item['source_text']) or normalize(item['source_text']) not in source):
            raise ValueError('A request constraint must quote its user source')
        field, op, value = item['field'], item['operator'], item['value']
        if text_values is not None and field in text_values:
            choices = {normalize(candidate): candidate for candidate in text_values[field]}
            if normalize(value) not in choices:
                raise ValueError('Unknown scalar category for ' + field + ': ' + value)
            value = choices[normalize(value)]
        if field in ('genres', 'tags'):
            choices = names.get(normalize(value), [])
            if not choices:
                raise ValueError('Unknown catalog feature: ' + value)
            field, value = next((row for row in choices if row[0] == field), choices[0])
            if op not in ('eq', 'ne', 'contains', 'not_contains'):
                raise ValueError('Feature constraints require membership operators')
            # Feature names are atomic, not substring searches.
            op = 'ne' if op in ('ne', 'not_contains') else 'eq'
        if re.search(r'(^|[._])(?:title|description|synopsis|summary|name)(?:[._]|$)', field.casefold()):
            quoted = any(mark + value + mark in question for mark in ('"', "'")) or '“' + value + '”' in question
            if not quoted:
                raise ValueError('Literal text filters require a quoted search string')
        constraint = {'field': field, 'operator': op, 'value': value}
        predicates([constraint], fields)  # Validate field/operator/value types before retrieval.
        if constraint not in compiled:
            compiled.append(constraint)
            mappings.append({**item, **constraint})
    preference_names = {normalize(f['value']) for f in compiled}
    entities = [query for query in result['entity_queries']
                if normalize(query) not in preference_names
                or any(mark + query + mark in question for mark in ('"', "'"))
                or '“' + query + '”' in question]
    return {**result, 'use_previous_context': result['use_previous_context'] and bool(history),
            'entity_queries': entities, 'filters': compiled, 'mappings': mappings}
