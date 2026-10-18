"""Render recommendations from verified catalog records without inventing constraints."""
import re


LABELS = {'year': 'release year', 'start_date': 'first release date', 'episodes': 'episode count', 'duration': 'minutes per episode',
          'source_material': 'source material'}
OPERATORS = {'eq': '', 'ne': 'excluding ', 'lt': 'under ', 'lte': 'at most ', 'gt': 'over ',
             'gte': 'at least ', 'contains': 'including ', 'not_contains': 'excluding '}


def render_recommendations(question, request, items):
    request = request or {}
    descriptions = []
    for field in request.get('filters', []):
        name, op, value = field['field'], field['operator'], field['value']
        if name in ('genres', 'tags'):
            descriptions.append(OPERATORS[op] + value)
        else:
            descriptions.append(LABELS.get(name, name.replace('_', ' ')) + ': ' + OPERATORS[op] + value)
    descriptions.extend(name.replace('_', ' ') + ': ' + str(value)
                        for name, value in request.get('structured_attributes', {}).items())
    intro = 'Catalog matches' + (' — ' + '; '.join(descriptions) if descriptions else '') + '.'
    if request.get('soft_query'):
        intro += ' Ranked by your stated preferences; mood and tone are approximate matches.'
    rows, seen = [], set()
    measured = {f['field'] for f in request.get('filters', [])}
    for item in items:
        identifier = str(item['id'])
        if identifier in seen:
            continue
        seen.add(identifier)
        description = re.sub(r'\s+', ' ', item.get('description', '')).strip()
        if len(description) > 450:
            description = description[:450].rsplit(' ', 1)[0].rstrip(' ,;:') + '…'
        metadata = []
        if 'start_date' in measured and item.get('start_date'):
            metadata.append(item['start_date'])
        for field, suffix in (('episodes', ' episodes'), ('duration', ' min/episode'), ('year', '')):
            value = item.get(field)
            if field in measured and isinstance(value, (int, float)):
                metadata.append(f'{value:g}' + suffix)
        title = '**' + item['title'] + '**' + (' (' + '; '.join(metadata) + ')' if metadata else '')
        rows.append(str(len(rows) + 1) + '. ' + title + (' — ' + description if description else ''))
    notes = list(request.get('limitations', []))
    for mapping in request.get('mappings', []):
        if mapping['mapping'] == 'broader' and mapping['field'] in ('genres', 'tags'):
            notes.append('A broader catalog category was used: ' + mapping['value'] + '.')
    if any(f['field'] in ('genres', 'tags') and f['operator'] in ('ne', 'not_contains') for f in request.get('filters', [])):
        notes.append('Excluded labels are absent from these catalog records; this does not guarantee that every scene '
                     'avoids the theme or setting.')
    return intro + '\n\n' + '\n\n'.join(rows) + ('\n\n' + ' '.join(dict.fromkeys(notes)) if notes else '')
