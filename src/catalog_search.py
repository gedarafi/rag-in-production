"""General field predicates for the imported catalog; no language intent rules."""
import math
import json
from datetime import date

from .anime_tools import normalize


BASE_FIELDS = {**dict.fromkeys(('title', 'description', 'format', 'status', 'source_material'), 'text'),
               'start_date': 'date',
               **dict.fromkeys(('episodes', 'duration', 'year', 'score', 'popularity'), 'number'),
               'adult': 'boolean',
               **dict.fromkeys(('aliases', 'genres', 'tags', 'studios', 'staff', 'characters'), 'collection')}
OPERATORS = ('eq', 'ne', 'lt', 'lte', 'gt', 'gte', 'contains', 'not_contains')
SORT_FIELDS = ('score', 'popularity', 'year', 'episodes', 'duration')


def preference_order(preferences, fields):
    """Rank typed preferred numeric ranges before applying the result limit."""
    clauses, params = [], []
    for preference in preferences:
        field, lo, hi = preference['field'], preference['min'], preference['max']
        if fields.get(field) != 'number' or any(type(v) not in (float, int) or not math.isfinite(v)
                                               for v in (lo, hi)) or lo > hi:
            raise ValueError('Invalid numeric preference')
        expression, args = field_expression(field, fields)
        clauses.append('CASE WHEN CAST(' + expression + ' AS REAL) BETWEEN ? AND ? THEN 0 ELSE 1 END,')
        params.extend([*args, lo, hi])
    return ''.join(clauses), params


def preference_key(record, preferences):
    return tuple(0 if isinstance(record.get(p['field']), (float, int)) and
                 p['min'] <= record[p['field']] <= p['max'] else 1 for p in preferences)


def evidence_value(value, needle):
    """Keep a large matching credit/relation list from consuming the answer window."""
    if len(json.dumps(value, ensure_ascii=False)) <= 1400:
        return value
    matches = []
    target = normalize(needle)
    def visit(item, path):
        if len(matches) >= 8:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, path + '.' + key)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, path + '.' + str(index))
        elif item is not None and target in normalize(str(item)):
            matches.append({'field': path.lstrip('.'), 'value': str(item)[:250]})
    visit(value, '')
    return {'matching_values': matches, 'other_values_omitted': True}


def field_expression(field, fields):
    if field not in fields:
        raise ValueError('Unknown search field: ' + field)
    if field == 'start_date':
        return ("(SELECT catalog_date(json_extract(m.data,'$.search_data.startDate_year'),"
                "json_extract(m.data,'$.search_data.startDate_month'),"
                "json_extract(m.data,'$.search_data.startDate_day')) "
                "FROM anime_metadata m WHERE m.anime_id=a.id)", [])
    if field == 'characters':
        return ("(SELECT json_group_array(json_object('name',json_extract(c.data,'$.name'),"
                "'aliases',json(json_extract(c.data,'$.aliases')),'description',json_extract(c.data,'$.description'),"
                "'role',p.role)) FROM appearances p JOIN characters c ON c.id=p.character_id WHERE p.anime_id=a.id)", [])
    if field in ('source_material', 'studios', 'staff'):
        return ("(SELECT json_extract(m.data,?) FROM anime_metadata m WHERE m.anime_id=a.id)", ['$.' + field])
    if field in BASE_FIELDS:
        return 'json_extract(a.data,?)', ['$.' + field]
    return ("(SELECT json_extract(m.data,?) FROM anime_metadata m WHERE m.anime_id=a.id)",
            ['$.search_data.' + field])


def predicates(filters, fields):
    if not isinstance(filters, list) or len(filters) > 12:
        raise ValueError('Use at most 12 field filters')
    clauses, params = [], []
    for item in filters:
        if not isinstance(item, dict) or set(item) != {'field', 'operator', 'value'}:
            raise ValueError('A filter needs field, operator and value')
        field, op, value = item['field'], item['operator'], item['value']
        if not all(isinstance(v, str) for v in (field, op, value)) or op not in OPERATORS:
            raise ValueError('Invalid field filter')
        expr, args = field_expression(field, fields)
        kind = fields[field]
        if kind == 'date':
            if op not in ('eq', 'ne', 'lt', 'lte', 'gt', 'gte'):
                raise ValueError('Date filters require calendar comparison operators')
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError('Noncanonical date')
            except ValueError as error:
                raise ValueError('ISO calendar date YYYY-MM-DD required for ' + field) from error
            symbol = {'eq': '=', 'ne': '!=', 'lt': '<', 'lte': '<=', 'gt': '>', 'gte': '>='}[op]
            clauses.append(f'{expr} {symbol} ?')
            params.extend([*args, value])
        elif op in ('lt', 'lte', 'gt', 'gte') or kind == 'number':
            if kind != 'number' or op in ('contains', 'not_contains'):
                raise ValueError('Numeric comparisons need a numeric field')
            try:
                value = float(value)
            except ValueError as error:
                raise ValueError('Numeric filter value required for ' + field) from error
            if not math.isfinite(value):
                raise ValueError('Finite numeric filter value required')
            symbol = {'eq': '=', 'ne': '!=', 'lt': '<', 'lte': '<=', 'gt': '>', 'gte': '>='}[op]
            clauses.append(f'{expr} {symbol} ?')
            params.extend([*args, value])
        elif kind == 'boolean':
            if op not in ('eq', 'ne') or value.lower() not in ('true', 'false'):
                raise ValueError('Boolean filters use eq/ne and true/false')
            clauses.append(f'{expr} {"=" if op == "eq" else "!="} ?')
            params.extend([*args, int(value.lower() == 'true')])
        elif kind == 'collection':
            if op not in ('eq', 'ne', 'contains', 'not_contains'):
                raise ValueError('Collections support eq/ne/contains/not_contains')
            test = 'normalize_text(j.atom)=?' if op in ('eq', 'ne') else 'instr(normalize_text(j.atom),?)>0'
            negative = op in ('ne', 'not_contains')
            clauses.append(f'{expr} IS NOT NULL AND {"NOT " if negative else ""}EXISTS '
                           f'(SELECT 1 FROM json_tree({expr}) j WHERE j.atom IS NOT NULL AND {test})')
            params.extend([*args, *args, normalize(value)])
        else:
            test = 'normalize_text(' + expr + ')'
            if op in ('contains', 'not_contains'):
                clauses.append(f'{expr} IS NOT NULL AND instr({test},?) {"=" if op == "not_contains" else ">"} 0')
                params.extend([*args, *args, normalize(value)])
            else:
                clauses.append(f'{expr} IS NOT NULL AND {test} {"=" if op == "eq" else "!="} ?')
                params.extend([*args, *args, normalize(value)])
    return clauses, params
