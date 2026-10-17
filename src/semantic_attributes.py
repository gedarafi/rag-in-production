"""Interpret natural-language measurements using small, catalog-grounded model calls."""
import json
import math
import re

from .anime_tools import normalize

ATTRIBUTE_DOCUMENTS = {
    'episodes': ['episodes', 'episode count number of episodes'],
    'duration': ['per episode runtime duration in minutes hours'],
    'year': ['release date year when released aired premiere'],
    'score': ['rating score rated average score'],
    'popularity': ['popularity audience size number of viewers'],
    'theme': ['anime genre theme romance action fantasy emotional story']}
BREVITY_DOCUMENTS = {
    'episodes': ['short', 'brief', 'compact series',
                 'short anime brief series compact show fewer episodes a short series one cour'],
    'duration': ['short episodes brief episodes few minutes each episode']}
PERIOD_DOCUMENTS = {'year': ['release decade era seventies eighties nineties 1980s 1990s 2000s']}

RANGE_SCHEMA = {'type': 'object', 'required': ['min', 'max'],
                'properties': {'min': {'type': ['number', 'null']}, 'max': {'type': ['number', 'null']}}}

COMPARISON_MEANINGS = {
    'gt': ('more than', 'over', 'greater than', 'after', 'later than'),
    'gte': ('at least', 'not less than', 'since', 'from'),
    'lt': ('less than', 'under', 'fewer than', 'before', 'earlier than'),
    'lte': ('at most', 'up to', 'no more than', 'until', 'through')}


def retrieve_comparison(embed, question, field, value, lower):
    """Resolve a comparison against semantic operator definitions, without text rules."""
    unit = {'year': 'release year', 'duration': 'minutes per episode', 'episodes': 'episodes'}.get(field, field)
    names = ('gt', 'gte') if lower else ('lt', 'lte')
    rendered = f'{value:g}'
    docs = [(op, phrase + ' ' + rendered + ' ' + unit) for op in names for phrase in COMPARISON_MEANINGS[op]]
    scores = embed([question])[0] @ embed([text for _, text in docs]).T
    return docs[max(range(len(docs)), key=lambda index: scores[index])][0]


def retrieve_attributes(embed, question, phrases, fields, features=()):
    """Retrieve schema meanings for request clauses; no lexical intent rules."""
    docs = [(name, text, 'range') for name, texts in ATTRIBUTE_DOCUMENTS.items() for text in texts
            if name == 'theme' or fields.get(name) == 'number']
    docs += [(name, text, 'brief') for name, texts in BREVITY_DOCUMENTS.items() for text in texts
             if fields.get(name) == 'number']
    docs += [(name, text, 'decade') for name, texts in PERIOD_DOCUMENTS.items() for text in texts
             if fields.get(name) == 'number']
    docs += [('theme', text, 'range') for kind, _, name in features if kind == 'genre'
             for text in (name, name + ' anime')]
    # Token spans recover properties omitted by generative decomposition. Their
    # meanings are still selected by embedding retrieval, never word rules.
    candidates = list(dict.fromkeys([question, *phrases, *question.split()]))
    vectors = embed([text for _, text, _ in docs])
    similarities = embed(candidates) @ vectors.T
    selected = {}
    for phrase, scores in zip(candidates, similarities):
        best = max(range(len(docs)), key=lambda index: scores[index])
        field = docs[best][0]
        mode = docs[best][2]
        # Retain local context when count and per-episode runtime compete.
        # These are ordinary text windows, not patterns for dates or lengths.
        if field in ('episodes', 'duration') and phrase != question:
            match = re.search(re.escape(phrase), question, re.I)
            if match:
                context = question[max(0, match.start() - 35):match.end() + 35]
                context_scores = embed([context])[0] @ vectors.T
                contextual = max(range(len(docs)), key=lambda index: context_scores[index])
                if docs[contextual][0] in ('episodes', 'duration') and docs[contextual][0] != field:
                    best, field = contextual, docs[contextual][0]
                    mode = docs[best][2]
        # Isolated tokens need stronger evidence than complete clauses. Common
        # words otherwise acquire arbitrary length meanings through weak neighbors.
        threshold = .65 if len(phrase.split()) == 1 and field != 'year' else .4
        if field != 'theme' and scores[best] >= threshold:
            selected.setdefault(field, {'sources': [], 'interpretations': []})['sources'].append(phrase)
            selected[field]['interpretations'].append(mode)
    return selected


NUMERIC_FIELDS = ('year', 'episodes', 'duration', 'score', 'popularity')
POLICIES = {
    'episodes': {'maximum': 50, 'preferred': (10, 19)},
    'duration': {'maximum': 15}}


def interpret_attributes(call, question, fields, meanings, retrieved, embed=None):
    filters, preferences, notes, sources = [], [], [], []
    if fields.get('start_date') == 'date':
        from .release_dates import release_date_filters
        date_filters, date_sources = release_date_filters(question)
        filters.extend(date_filters)
        sources.extend(date_sources)
        if date_filters:
            sources.extend(retrieved.get('year', {}).get('sources', []))
    for field in NUMERIC_FIELDS:
        if field == 'year' and any(f['field'] == 'start_date' for f in filters):
            continue  # Full calendar bounds supersede a lossy year interpretation.
        if fields.get(field) != 'number':
            continue
        if field not in retrieved:
            continue
        evidence = retrieved[field]
        field_sources = evidence['sources']
        source = min(field_sources, key=len)
        brief = all(mode == 'brief' for mode in evidence['interpretations'])
        decade = field == 'year' and 'decade' in evidence['interpretations']
        # The model chooses the interpretation; Python enforces the product policy.
        if brief:
            policy = POLICIES[field]
            bounds = [{'operator': 'lte', 'value': policy['maximum']}]
            if 'preferred' in policy:
                lo, hi = policy['preferred']
                preferences.append({'field': field, 'min': lo, 'max': hi})
            note = 'Interpreting brevity as at most ' + str(policy['maximum']) + ' ' + (
                'episodes' if field == 'episodes' else 'minutes per episode') + '.'
            if 'preferred' in policy:
                note += ' Prioritizing titles with 10–19 episodes; this is a preference, not a strict category.'
            notes.append(note)
        else:
            unit = 'release years' if field == 'year' else meanings[field]
            if decade:
                period = min((text for text, mode in zip(field_sources, evidence['interpretations']) if mode == 'decade'), key=len)
                result = call('Interpret the release period ' + json.dumps(period) +
                              '. Return the inclusive start and end years as min and max. A decade covers ten years.',
                              {'type': 'object', 'required': ['min', 'max'], 'properties': {
                                  'min': {'type': 'number'}, 'max': {'type': 'number'}}})
                if embed:
                    relations = ['anime released during ' + period, 'anime released before ' + period,
                                 'anime released after ' + period]
                    scores = embed([question])[0] @ embed(relations).T
                    relation = max(range(len(relations)), key=lambda index: scores[index])
                    if relation == 1:
                        result = {'min': None, 'max': result['min']}
                    elif relation == 2:
                        result = {'min': result['max'], 'max': None}
                else:
                    relation = 0
            else:
                result = call('Interpret the requested ' + unit + ' in ' + json.dumps(question) + '. '
                          'Return the lower and upper bounds as min and max. Use null for a bound that was '
                          'not requested. Do not cap an open range at the current year. ' +
                          ('A decade covers ten years.' if field == 'year' else
                           'Convert hours to minutes.' if field == 'duration' else '') +
                          ' Respect negation.', RANGE_SCHEMA)
            lo, hi = result.get('min'), result.get('max')
            for value in (lo, hi):
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError('Invalid interpreted numeric boundary')
            if lo is not None and hi is not None and lo > hi:
                raise ValueError('Interpreted numeric bounds are contradictory')
            bounds = []
            for key, inclusive, exclusive in [('min', 'gte', 'gt'), ('max', 'lte', 'lt')]:
                if result.get(key) is not None:
                    if decade:
                        operator = inclusive if relation == 0 else exclusive
                    elif embed:
                        operator = retrieve_comparison(embed, question, field, result[key], key == 'min')
                    else:
                        operator = call('For the request ' + json.dumps(question) + ', interpret the ' + key +
                                        ' boundary ' + str(result[key]) + ' of ' + meanings[field] + '. Use ' +
                                        inclusive + ' if the boundary is included, ' + exclusive + ' if excluded.',
                                        {'type': 'object', 'required': ['operator'],
                                         'properties': {'operator': {'enum': [inclusive, exclusive]}}})['operator']
                    # Zero is the natural lower domain of counts and runtimes,
                    # not an additional requested restriction.
                    if key == 'min' and result[key] == 0 and hi is not None and hi > 0:
                        continue
                    bounds.append({'operator': operator, 'value': result[key]})
            if result.get('min') is not None and result.get('min') == result.get('max'):
                bounds = [{'operator': 'eq', 'value': result['min']}]
        if not source or source.casefold() not in question.casefold():
            raise ValueError('A measurable property must quote its user source')
        if not bounds:
            raise ValueError('Cannot interpret measurable requirement: ' + source)
        sources.extend(field_sources)
        for bound in bounds:
            value = bound['value']
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid interpreted numeric boundary')
            if field in ('year', 'episodes') and int(value) != value:
                raise ValueError('Year and episode bounds must be integers')
            filters.append({'field': field, 'operator': bound['operator'], 'value': str(value).removesuffix('.0'),
                            'source_text': source, 'mapping': 'broader' if brief else 'exact'})
    return filters, preferences, notes, sources


def covered_property(phrase, sources, stop):
    """Exclude interpreted property phrases from title/theme resolution."""
    remaining = normalize(phrase)
    for source in sorted(sources, key=len, reverse=True):
        remaining = remaining.replace(normalize(source), ' ')
    return not (set(remaining.split()) - stop)
