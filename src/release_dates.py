"""Calendar values and explicit ISO-date boundaries, independent of LLM guesses."""
from datetime import date
import re

ISO = r'(?<![\w-])\d{4}-\d{2}-\d{2}(?![\w-])'


def complete_date(year, month, day):
    """Unknown or invalid date parts never become invented January/day-one dates."""
    try:
        parts = [int(v) for v in (year, month, day)]
        if any(isinstance(v, bool) or float(v) != part for v, part in zip((year, month, day), parts)):
            return None
        return date(*parts).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def release_date_filters(question):
    """Parse explicitly bounded ISO dates without projecting them to calendar years.

    This handles calendar notation and comparison operators only; semantic genre,
    mood, year-only, and decade interpretation remain in their existing resolver.
    """
    dates = list(re.finditer(ISO, question))
    if not dates:
        return [], []
    for match in dates:
        date.fromisoformat(match[0])  # reject impossible calendar values
    filters, sources, used = [], [], set()

    def add(operator, value, source):
        item = {'field': 'start_date', 'operator': operator, 'value': value,
                'source_text': source, 'mapping': 'exact'}
        if item not in filters:
            filters.append(item)
        sources.extend([source, value])

    ranges = re.compile(r'(?P<lo>' + ISO + r')\s*(?:through|to|until|and|[–—])\s*(?P<hi>' + ISO + r')', re.I)
    for match in ranges.finditer(question):
        # "after A and before B" is handled by its individual operators below.
        lo, hi = match['lo'], match['hi']
        if re.search(r'\b(?:not\s+(?:released\s+)?(?:from|between)|outside|excluding|except|without)\s*$',
                     question[:match.start()], re.I):
            raise ValueError('Excluded release-date ranges require an outside-range query; they cannot be treated as an included range.')
        if lo > hi:
            raise ValueError('Release date range is contradictory; start date must not exceed end date.')
        exclusive = bool(re.match(r'\s*\(?exclusive\b', question[match.end():], re.I))
        strict_lower = bool(re.search(r'(?<!not )\b(?:after|later\s+than)\s*$', question[:match.start()], re.I))
        add('gt' if exclusive or strict_lower else 'gte', lo, match[0])
        add('lt' if exclusive else 'lte', hi, match[0])
        used.update((match.start('lo'), match.start('hi')))
    operators = [
        ('gte', r'on\s+or\s+after|not\s+before|no\s+earlier\s+than|since|from|starting(?:\s+on)?'),
        ('lte', r'on\s+or\s+before|not\s+after|no\s+later\s+than|through|until|up\s+to'),
        ('gt', r'after|later\s+than'), ('lt', r'before|earlier\s+than'),
        ('ne', r'not\s+on|except(?:\s+on)?'),
        ('eq', r'on|exactly')]
    for match in dates:
        if match.start() in used:
            continue
        prefix = question[:match.start()]
        for operator, words in operators:
            found = re.search(r'\b(?:' + words + r')\s*$', prefix, re.I)
            if found:
                add(operator, match[0], question[found.start():match.end()])
                used.add(match.start())
                break
    if len(used) != len(dates):
        raise ValueError('Specify release dates using from/through, between/and, on, before, or after.')
    lower = [(f['value'], f['operator']) for f in filters if f['operator'] in ('gt', 'gte', 'eq')]
    upper = [(f['value'], f['operator']) for f in filters if f['operator'] in ('lt', 'lte', 'eq')]
    if lower and upper:
        lo, hi = max(v for v, _ in lower), min(v for v, _ in upper)
        if lo > hi or (lo == hi and (any(v == lo and op == 'gt' for v, op in lower) or
                                    any(v == hi and op == 'lt' for v, op in upper))):
            raise ValueError('Release date boundaries have no possible matching date.')
    return filters, list(dict.fromkeys(sources))
