"""Bounded tool plans with references to earlier results, without LLM replanning."""
from __future__ import annotations

import json
import re
from .catalog_search import OPERATORS, SORT_FIELDS


def plan_schema(max_steps, searchable_fields=None):
    scalar = {'type': 'string', 'maxLength': 1500}
    ids = {'type': 'array', 'maxItems': 10, 'items': scalar}
    fields = {
        'kind': {'enum': ['anime', 'character', 'genre', 'tag']}, 'query': scalar, 'id': scalar,
        'section': {'enum': ['overview', 'cast', 'staff', 'studios', 'relations', 'voices', 'appearances']},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
        'year': {'type': 'integer'}, 'format': scalar,
        'year_from': {'type': 'integer', 'minimum': 1, 'maximum': 9999,
                      'description': 'Inclusive lower release-year bound, ONLY if requested. Omit for before/until requests.'},
        'year_to': {'type': 'integer', 'minimum': 1, 'maximum': 9999,
                    'description': 'Inclusive upper release-year bound, ONLY if requested. Omit for after/since requests.'},
        'studio': {**scalar, 'description': 'Explicitly requested studio name. Do not infer a studio from a staff member.'},
        'staff': {**scalar, 'description': 'Named person whose staff credit must match, e.g. Hayao Miyazaki. Not an anime ID.'},
        'staff_role': {**scalar, 'description': 'Exact credited role for that same person, e.g. Director or Music.'},
        'source_material': scalar,
        'sort_by': {'enum': list(SORT_FIELDS)},
        'sort_order': {'enum': ['asc', 'desc']},
        'filters': {'type': 'array', 'maxItems': 12, 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['field', 'operator', 'value'],
            'properties': {'field': ({'enum': sorted(searchable_fields)} if searchable_fields else scalar),
                           'operator': {'enum': list(OPERATORS)},
                           'value': {**scalar, 'description': 'Numeric fields use numeric strings; start_date uses full YYYY-MM-DD dates. Never put a date string in year.'}}}},
        'tag_match': {'enum': ['all', 'any']},
        **{key: ids for key in ('anime_ids', 'seed_anime_ids', 'character_ids', 'genre_ids', 'tag_ids')},
    }
    metadata_filters = ['year', 'year_from', 'year_to', 'format', 'studio', 'staff', 'staff_role', 'source_material', 'sort_by', 'sort_order', 'filters']
    signatures = {
        'search_features': (['kind', 'query', 'limit'], ['query']),
        'resolve_entity': (['kind', 'query', 'anime_ids', 'limit'], ['kind', 'query']),
        'get_anime': (['id', 'section', 'offset', 'limit'], ['id']),
        'get_character': (['id', 'section', 'offset', 'limit'], ['id']),
        'recommend_anime': (['seed_anime_ids', 'genre_ids', 'tag_ids', 'tag_match', 'query', 'limit', *metadata_filters], []),
        'recommend_by_character': (['character_ids', 'query', 'limit'], ['character_ids', 'query']),
        'popular': (['genre_ids', 'tag_ids', 'limit', *metadata_filters], []),
        'search_passages': (['query', 'anime_ids', 'limit'], ['query']),
    }
    steps = []
    for name, (allowed, required) in signatures.items():
        steps.append({'type': 'object', 'additionalProperties': False,
                      'required': ['tool', 'arguments'], 'properties': {
                          'tool': {'const': name}, 'arguments': {
                              'type': 'object', 'additionalProperties': False,
                              'properties': {key: ({'enum': ['genre', 'tag']} if name == 'search_features' and key == 'kind'
                                                   else fields[key]) for key in allowed}, 'required': required}}})
    normal = {'type': 'object', 'additionalProperties': False,
            'required': ['scope', 'direct_response', 'use_previous_context', 'goal', 'clarification', 'steps'], 'properties': {
                'scope': {'type': 'string', 'enum': ['in_domain', 'uncertain']},
                'direct_response': {'const': ''},
                'use_previous_context': {'type': 'boolean'},
                'goal': {'type': 'string', 'maxLength': 1000},
                'clarification': {'type': 'string', 'maxLength': 500},
                'steps': {'type': 'array', 'maxItems': max_steps, 'items': {'anyOf': steps}}}}
    direct = {'type': 'object', 'additionalProperties': False, 'required': normal['required'],
              'properties': {'scope': {'const': 'out_of_domain'},
                             'direct_response': {'type': 'string', 'minLength': 1, 'maxLength': 1500},
                             'use_previous_context': {'const': False}, 'goal': {'const': ''},
                             'clarification': {'const': ''},
                             'steps': {'type': 'array', 'maxItems': 0, 'items': {'type': 'null'}}}}
    # Separate complete object branches let Ollama's grammar enforce that an
    # out-of-domain reply cannot also request tools. No extra model call.
    return {'anyOf': [direct, normal]}


def validate_plan(raw, schema):
    tool_schema = schema['anyOf'][1]
    plan = json.loads(raw)
    if not isinstance(plan, dict) or set(plan) != {'scope', 'direct_response', 'use_previous_context', 'goal', 'clarification', 'steps'}:
        raise ValueError('Plan requires scope, direct_response, use_previous_context, goal, clarification and steps')
    if not isinstance(plan['scope'], str) or plan['scope'] not in ('in_domain', 'out_of_domain', 'uncertain'):
        raise ValueError('Invalid request scope')
    if not isinstance(plan['direct_response'], str) or len(plan['direct_response']) > 1500:
        raise ValueError('Invalid direct response')
    if type(plan['use_previous_context']) is not bool:
        raise ValueError('use_previous_context must be a boolean')
    if not isinstance(plan['goal'], str) or len(plan['goal']) > 1000:
        raise ValueError('Invalid response goal')
    if not isinstance(plan['clarification'], str) or len(plan['clarification']) > 500:
        raise ValueError('Invalid clarification guidance')
    steps = plan['steps']
    if not isinstance(steps, list) or len(steps) > tool_schema['properties']['steps']['maxItems']:
        raise ValueError('Invalid number of planned functions')
    definitions = {b['properties']['tool']['const']: b['properties']['arguments']
                   for b in tool_schema['properties']['steps']['items']['anyOf']}
    for step in steps:
        if not isinstance(step, dict) or set(step) != {'tool', 'arguments'} or step['tool'] not in definitions:
            raise ValueError('Invalid planned function')
        args, definition = step['arguments'], definitions[step['tool']]
        if not isinstance(args, dict) or set(args) - set(definition['properties']) or set(definition['required']) - set(args):
            raise ValueError('Invalid arguments for ' + step['tool'])
        for key, value in args.items():
            field = definition['properties'][key]
            if field.get('type') == 'integer':
                if type(value) is not int or value < field.get('minimum', -100000) or value > field.get('maximum', 100000):
                    raise ValueError('Invalid integer argument: ' + key)
            elif field.get('type') == 'array':
                if key == 'filters':
                    if not isinstance(value, list) or len(value) > 12:
                        raise ValueError('Invalid field filters')
                    for item in value:
                        if (not isinstance(item, dict) or set(item) != {'field', 'operator', 'value'}
                                or any(not isinstance(v, str) or len(v) > 1500 for v in item.values())
                                or item['operator'] not in OPERATORS):
                            raise ValueError('Invalid field filter')
                        allowed_fields = field['items']['properties']['field'].get('enum')
                        if allowed_fields and item['field'] not in allowed_fields:
                            raise ValueError('Unknown search field: ' + item['field'])
                    continue
                if not isinstance(value, list) or len(value) > 10 or any(not isinstance(v, str) or len(v) > 1500 for v in value):
                    raise ValueError('Invalid ID list: ' + key)
            elif not isinstance(value, str) or len(value) > 1500 or 'enum' in field and value not in field['enum']:
                raise ValueError('Invalid string argument: ' + key)
    return plan


def resolve_references(value, results):
    """$0.matches.0.id reads step zero; * maps a field over a result list."""
    if isinstance(value, str) and value.startswith('$'):
        match = re.fullmatch(r'\$(\d+)((?:\.(?:[A-Za-z_]\w*|\d+|\*))*)', value)
        if not match or int(match[1]) >= len(results) or results[int(match[1])] is None:
            raise ValueError('References must address an earlier successful step: ' + value)
        result = results[int(match[1])]
        if result.get('status') == 'error':
            raise ValueError('A required earlier step failed: ' + value)
        def walk(item, path):
            if not path:
                return item
            key, *tail = path
            if key == '*':
                if not isinstance(item, list) or len(item) > 50:
                    raise ValueError('Wildcard requires a bounded result list')
                return [walk(child, tail) for child in item]
            if isinstance(item, list) and key.isdigit():
                if int(key) >= len(item):
                    raise ValueError('No matching entity for required result reference: ' + value)
                return walk(item[int(key)], tail)
            if isinstance(item, dict) and key in item:
                return walk(item[key], tail)
            raise ValueError('Missing result reference: ' + value)
        return walk(result, match[2].split('.')[1:])
    if isinstance(value, list):
        output = []
        for item in value:
            resolved = resolve_references(item, results)
            if isinstance(item, str) and item.startswith('$') and resolved == []:
                raise ValueError('Required result reference matched no items: ' + item)
            output.extend(resolved if isinstance(resolved, list) else [resolved])
        return output
    if isinstance(value, dict):
        return {key: resolve_references(child, results) for key, child in value.items()}
    return value


def compile_steps(steps):
    """Schedule declared dependencies and repair unambiguous result-type mixups.

    A matches path must come from a dictionary, for example. This never adds a
    tool or selects an entity; it only binds the functions the LLM already chose.
    """
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    compiled = json.loads(json.dumps(steps))
    def rewrite(value):
        if isinstance(value, str):
            dictionary_id = re.fullmatch(r'\$(\d+)\.id', value)
            if dictionary_id and int(dictionary_id[1]) < len(steps) and steps[int(dictionary_id[1])]['tool'] in ('resolve_entity', 'search_features'):
                return '$' + dictionary_id[1] + '.matches.0.id'
            match = re.fullmatch(r'\$(\d+)\.(matches|main_characters|cast)(\..*)?', value)
            if match:
                index, root = int(match[1]), match[2]
                producers = ('resolve_entity', 'search_features') if root == 'matches' else ('get_anime',)
                if index < len(steps) and steps[index]['tool'] not in producers:
                    candidates = [i for i, step in enumerate(steps) if step['tool'] in producers]
                    if len(candidates) == 1:
                        return '$' + str(candidates[0]) + '.' + root + (match[3] or '')
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value
    for step in compiled:
        step['arguments'] = rewrite(step['arguments'])
    for index, step in enumerate(compiled):
        args = step['arguments']
        if step['tool'] == 'recommend_anime' and args.get('tag_match') == 'any' and not args.get('tag_ids'):
            # Repair only an unambiguous missing dataflow link. The model chose
            # the semantic tag search and alternative-tag recommendation itself.
            searches = [i for i in range(index) if compiled[i]['tool'] == 'search_features'
                        and compiled[i]['arguments'].get('kind', 'tag') == 'tag']
            if len(searches) == 1:
                args['tag_ids'] = [f'${searches[0]}.matches.*.id']
    dependencies = []
    for step in compiled:
        dependencies.append({int(m[1]) for text in strings(step['arguments'])
                             if (m := re.match(r'^\$(\d+)(?:\.|$)', text)) and int(m[1]) < len(steps)})
    pending, order = set(range(len(steps))), []
    while pending:
        ready = sorted(i for i in pending if not dependencies[i] & pending)
        if not ready:
            # Cyclic references remain unavailable and become structured failures
            # at execution, rather than blocking unrelated successful results.
            order.extend(sorted(pending))
            break
        order.extend(ready)
        pending.difference_update(ready)
    return compiled, order
