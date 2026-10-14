"""Bounded tool plans with references to earlier results, without LLM replanning."""
from __future__ import annotations

import json
import re


def plan_schema(max_steps):
    scalar = {'type': 'string', 'maxLength': 1500}
    ids = {'type': 'array', 'maxItems': 10, 'items': scalar}
    fields = {
        'kind': {'enum': ['anime', 'character', 'genre', 'tag']}, 'query': scalar, 'id': scalar,
        'section': {'enum': ['overview', 'cast', 'staff', 'studios', 'relations', 'voices', 'appearances']},
        'offset': {'type': 'integer', 'minimum': 0},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
        'year': {'type': 'integer'}, 'format': scalar,
        **{key: ids for key in ('anime_ids', 'seed_anime_ids', 'character_ids', 'genre_ids', 'tag_ids')},
    }
    signatures = {
        'resolve_entity': (['kind', 'query', 'anime_ids', 'limit'], ['kind', 'query']),
        'get_anime': (['id', 'section', 'offset', 'limit'], ['id']),
        'get_character': (['id', 'section', 'offset', 'limit'], ['id']),
        'recommend_anime': (['seed_anime_ids', 'genre_ids', 'tag_ids', 'query', 'limit'], []),
        'recommend_by_character': (['character_ids', 'query', 'limit'], ['character_ids', 'query']),
        'popular': (['genre_ids', 'tag_ids', 'year', 'format', 'limit'], []),
        'search_passages': (['query', 'anime_ids', 'limit'], ['query']),
    }
    steps = []
    for name, (allowed, required) in signatures.items():
        steps.append({'type': 'object', 'additionalProperties': False,
                      'required': ['tool', 'arguments'], 'properties': {
                          'tool': {'const': name}, 'arguments': {
                              'type': 'object', 'additionalProperties': False,
                              'properties': {key: fields[key] for key in allowed}, 'required': required}}})
    return {'type': 'object', 'additionalProperties': False,
            'required': ['use_previous_context', 'goal', 'clarification', 'steps'], 'properties': {
                'use_previous_context': {'type': 'boolean'},
                'goal': {'type': 'string', 'maxLength': 1000},
                'clarification': {'type': 'string', 'maxLength': 500},
                'steps': {'type': 'array', 'maxItems': max_steps, 'items': {'anyOf': steps}}}}


def validate_plan(raw, schema):
    plan = json.loads(raw)
    if not isinstance(plan, dict) or set(plan) != {'use_previous_context', 'goal', 'clarification', 'steps'}:
        raise ValueError('Plan requires use_previous_context, goal, clarification and steps')
    if type(plan['use_previous_context']) is not bool:
        raise ValueError('use_previous_context must be a boolean')
    if not isinstance(plan['goal'], str) or len(plan['goal']) > 1000:
        raise ValueError('Invalid response goal')
    if not isinstance(plan['clarification'], str) or len(plan['clarification']) > 500:
        raise ValueError('Invalid clarification guidance')
    steps = plan['steps']
    if not isinstance(steps, list) or len(steps) > schema['properties']['steps']['maxItems']:
        raise ValueError('Invalid number of planned functions')
    definitions = {b['properties']['tool']['const']: b['properties']['arguments']
                   for b in schema['properties']['steps']['items']['anyOf']}
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
                return walk(item[int(key)], tail)
            if isinstance(item, dict) and key in item:
                return walk(item[key], tail)
            raise ValueError('Missing result reference: ' + value)
        return walk(result, match[2].split('.')[1:])
    if isinstance(value, list):
        output = []
        for item in value:
            resolved = resolve_references(item, results)
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
            if dictionary_id and int(dictionary_id[1]) < len(steps) and steps[int(dictionary_id[1])]['tool'] == 'resolve_entity':
                return '$' + dictionary_id[1] + '.matches.0.id'
            match = re.fullmatch(r'\$(\d+)\.(matches|main_characters|cast)(\..*)?', value)
            if match:
                index, root = int(match[1]), match[2]
                producer = 'resolve_entity' if root == 'matches' else 'get_anime'
                if index < len(steps) and steps[index]['tool'] != producer:
                    candidates = [i for i, step in enumerate(steps) if step['tool'] == producer]
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
