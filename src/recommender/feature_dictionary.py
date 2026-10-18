"""Cache the imported AniList tag definitions used for semantic preferences."""
import csv
import json
import io
import sqlite3
from contextlib import closing

from .anime_tools import normalize
from .sqlite_catalog import BUILD_LOCK


def feature_definitions(settings, catalog):
    # Fixture/custom catalogs must not accidentally consume another library's CSV.
    if settings.documents.resolve() != (settings.root / 'data/documents/anilist').resolve():
        return {}
    paths = [settings.root / folder / 'anilist_anime_data_complete.csv' for folder in ('data/raw', 'data-raw')]
    source = next((p for p in paths if p.is_file()), None)
    with BUILD_LOCK, closing(sqlite3.connect(catalog.database, timeout=120)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS entity_feature_info(id TEXT PRIMARY KEY,description TEXT,category TEXT)')
        stamp = f'{source.stat().st_mtime_ns}:{source.stat().st_size}' if source else 'unavailable'
        current = db.execute("SELECT value FROM meta WHERE key='feature_definitions'").fetchone()
        if source and current != (stamp,):
            definitions = {}
            csv.field_size_limit(max(csv.field_size_limit(), 100000000))
            from .datasets import read_anilist_csv
            with io.StringIO(read_anilist_csv(source)) as file:
                for row in csv.DictReader(file):
                    try:
                        tags = json.loads(row.get('tags') or '[]')
                    except (ValueError, TypeError):
                        continue
                    for tag in tags if isinstance(tags, list) else []:
                        if isinstance(tag, dict) and tag.get('name') and tag.get('description'):
                            definitions[normalize(tag['name'])] = (str(tag['description']), str(tag.get('category', '')))
            db.executemany('INSERT OR REPLACE INTO entity_feature_info VALUES (?,?,?)',
                           [(key, *values) for key, values in definitions.items()])
            db.execute("INSERT OR REPLACE INTO meta VALUES ('feature_definitions',?)", (stamp,))
            db.commit()
        return {identifier: {'description': description, 'category': category}
                for identifier, description, category in db.execute('SELECT id,description,category FROM entity_feature_info')}
