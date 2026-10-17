"""Bounded arithmetic for model-authored answer templates, independent of domain."""
import ast
from decimal import Decimal, DecimalException, localcontext
import re


def evaluate_quantity(expression, measurements, earlier=None):
    """Evaluate dimensions as well as values; source fields supply the units."""
    if not isinstance(expression, str) or len(expression) > 500:
        raise ValueError('Invalid quantity expression')
    unit_names = {'minutes', 'minute', 'hours', 'hour', 'days', 'day', 'seconds', 'second'}
    unit_names.update(unit for row in measurements.values() for unit in row['units'])
    unit_names.update(unit + 's' for unit in list(unit_names) if not unit.endswith('s'))
    # Some small models append a unit annotation despite the separate unit
    # field. Remove only recognized annotations; never execute other text.
    annotation = '|'.join(re.escape(unit) for unit in sorted(unit_names, key=len, reverse=True))
    expression = re.sub(r'\s+(?:' + annotation + r')(?:\s+per\s+(?:' + annotation + r'))?\s*$', '', expression)
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ValueError('Invalid quantity expression') from error
    if len(list(ast.walk(tree))) > 100:
        raise ValueError('Quantity expression is too complex')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(str(node.value))
            matches = [row for row in measurements.values() if Decimal(str(row['value'])) == value]
            dimensions = {tuple(sorted(row['units'].items())) for row in matches}
            if len(dimensions) > 1:
                raise ValueError('Ambiguous literal: use a measurement variable instead')
            return value, dict(next(iter(dimensions))) if dimensions else {}, bool(matches)
        if isinstance(node, ast.Name):
            if node.id in measurements:
                row = measurements[node.id]
                return Decimal(str(row['value'])), row['units'], True
            if node.id in (earlier or {}):
                return earlier[node.id]
            raise ValueError('Unknown quantity variable')
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value, units, sourced = visit(node.operand)
            return value * (-1 if isinstance(node.op, ast.USub) else 1), units, sourced
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, lu, ls = visit(node.left)
            right, ru, rs = visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if lu != ru:
                    raise ValueError('Cannot add or subtract quantities with different units')
                return (left + right if isinstance(node.op, ast.Add) else left - right), lu, ls or rs
            division = isinstance(node.op, ast.Div)
            units = dict(lu)
            for unit, exponent in ru.items():
                units[unit] = units.get(unit, 0) + (-exponent if division else exponent)
            return (left / right if division else left * right), {k: v for k, v in units.items() if v}, ls or rs
        raise ValueError('Unsupported quantity expression')
    try:
        with localcontext() as context:
            context.prec = 30
            result = visit(tree.body)
        if not result[0].is_finite() or abs(result[0]) > Decimal('1e18'):
            raise ValueError('Quantity outside supported bounds')
        if measurements and not result[2]:
            raise ValueError('A factual quantity must reference source measurements')
        return result
    except DecimalException as error:
        raise ValueError('Invalid quantity operation') from error


def render_quantities(items, measurements):
    if not isinstance(items, list) or not 1 <= len(items) <= 12:
        raise ValueError('Invalid quantity answer')
    earlier, lines, checks = {}, [], []
    for index, item in enumerate(items):
        if set(item) != {'label', 'expression', 'unit', 'decimals'} or not isinstance(item['label'], str):
            raise ValueError('Invalid quantity item')
        if type(item['decimals']) is not int or not 0 <= item['decimals'] <= 6:
            raise ValueError('Invalid quantity precision')
        value, units, sourced = evaluate_quantity(item['expression'], measurements, earlier)
        earlier['c' + str(index)] = (value, units, sourced)
        scales = {'minutes': Decimal(1), 'hours': Decimal(60), 'days': Decimal(1440), 'seconds': Decimal(1) / 60}
        if units == {'minute': 1} and item['unit'] in scales:
            value /= scales[item['unit']]
            label = item['unit']
        else:
            # The source dimension wins over an incompatible model unit label.
            names = {'minute': 'minutes', 'episode': 'episodes', 'year': 'years', 'character': 'characters'}
            positive = [names.get(k, k) + (f'^{v}' if v != 1 else '') for k, v in units.items() if v > 0]
            negative = [k + (f'^{abs(v)}' if v != -1 else '') for k, v in units.items() if v < 0]
            label = ' * '.join(positive) + (' per ' + ' * '.join(negative) if negative else '')
        formatted = format(value, f".{item['decimals']}f")
        if item['decimals']:
            formatted = formatted.rstrip('0').rstrip('.')
        lines.append(item['label'].rstrip(': ') + ': ' + formatted + (' ' + label if label else '') + '.')
        checks.append({'expression': item['expression'], 'value': formatted, 'unit': label})
    return '\n'.join(lines), checks
