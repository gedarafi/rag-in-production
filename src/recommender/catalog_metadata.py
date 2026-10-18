"""Searchable production metadata, upgraded from existing exports without embeddings."""
from contextlib import closing
import json
import math
import re
import sqlite3

from .anime_tools import normalize

# Markdown descriptions can contain lines such as "Director: ...". Only real
# export headers may become fields when upgrading legacy documents.
EXPORT_FIELDS = set('id idMal title title_romaji title_english title_native title_userPreferred type format status '
    'description startDate_year startDate_month startDate_day endDate_year endDate_month endDate_day season '
    'seasonYear seasonInt episodes duration countryOfOrigin isLicensed source trailer_id trailer_site trailer_thumbnail '
    'updatedAt coverImage_extraLarge coverImage_large coverImage_medium coverImage_color bannerImage genres synonyms '
    'tags averageScore meanScore popularity favourites trending rankings isFavourite isAdult isLocked siteUrl '
    'externalLinks streamingEpisodes relations characters staff studios airingSchedule recommendations reviews '
    'stats_scoreDistribution stats_statusDistribution nextAiringEpisode hashtag'.split())


def search_attributes(row):
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        return value
    omitted = {'description', 'characters', 'staff', 'studios', 'genres', 'tags', 'source'}
    result = {}
    for key, value in row.items():
        if key in omitted or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', key):
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
        result[key] = safe(value)
    def array(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return []
        return value if isinstance(value, list) else []
    voices = []
    for character in array(row.get('characters')):
        for person in character.get('voiceActors', []):
            voices.append({k: person[k] for k in ('id', 'name', 'languageV2') if k in person})
    if voices:
        result['voice_actors'] = voices
    if row.get('tags'):
        result['tag_details'] = array(row['tags'])
    return result


def production_metadata(row):
    def array(key):
        value = row.get(key) or []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return []
        return value if isinstance(value, list) else []
    studios, staff = [], []
    for edge in array('studios'):
        node = edge.get('node') or edge
        name = node.get('name')
        if isinstance(name, str) and name.strip():
            studios.append({'name': name, 'key': normalize(name),
                            'main': bool(edge.get('isMain')), 'animation': bool(node.get('isAnimationStudio'))})
    for edge in array('staff'):
        node = edge.get('node') or edge
        name = node.get('name') or {}
        if isinstance(name, str):
            name = {'full': name}
        aliases = [name.get('full'), name.get('native'), *(name.get('alternative') or [])]
        aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
        if aliases:
            keys = {normalize(a) for a in aliases}
            keys |= {' '.join(sorted(k.split())) for k in list(keys)}
            role = edge.get('role') or ''
            staff.append({'name': aliases[0], 'keys': sorted(keys), 'role': role, 'role_key': normalize(role)})
    return {'metadata_schema': 2, 'source_material': str(row.get('source') or '').strip(),
            'studios': studios, 'staff': staff, 'search_data': search_attributes(row)}


def ensure_production_metadata(folder, database, lock):
    """One-time SQL side-table upgrade; leaves documents and vector indexes intact."""
    with lock, closing(sqlite3.connect(database, timeout=120)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS anime_metadata(anime_id TEXT PRIMARY KEY,data TEXT NOT NULL)')
        stamp = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()[0] + ':production-v2'
        if db.execute("SELECT value FROM meta WHERE key='production_metadata'").fetchone() == (stamp,):
            return
        with db:
            db.execute('DELETE FROM anime_metadata')
            fields = {}
            for aid, data in db.execute('SELECT id,data FROM anime').fetchall():
                record = json.loads(data)
                raw = {}
                if record.get('metadata_schema') != 2:
                    path = folder / record['source']
                    if path.is_file():
                        with path.open(encoding='utf-8') as file:
                            for line in file:
                                key, _, value = line.partition(':')
                                if key in EXPORT_FIELDS:
                                    raw[key] = value.strip()
                if record.get('metadata_schema') in (1, 2):
                    metadata = {k: record.get(k) for k in ('source_material', 'studios', 'staff')}
                    metadata['search_data'] = record.get('search_data', search_attributes(raw))
                else:
                    metadata = production_metadata(raw)
                for key, value in metadata['search_data'].items():
                    if value is not None and value != '':
                        fields.setdefault(key, 'boolean' if isinstance(value, bool) else 'number' if isinstance(value, (int, float))
                                          else 'collection' if isinstance(value, (list, dict)) else 'text')
                db.execute('INSERT INTO anime_metadata VALUES (?,?)', (aid, json.dumps(metadata, ensure_ascii=False)))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('search_fields',?)", (json.dumps(fields),))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('production_metadata',?)", (stamp,))
