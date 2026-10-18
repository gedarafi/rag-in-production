"""Separate, searchable entity dictionaries and paged canonical bundles."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from difflib import SequenceMatcher

from .anime_tools import normalize, contains
from .sqlite_catalog import BUILD_LOCK
from .sqlite_catalog import STOP

KINDS = ('anime', 'character', 'genre', 'tag')


class EntityStore:
    def __init__(self, catalog):
        self.catalog = catalog
        self.database = catalog.database

    def ensure_dictionary(self):
        """Build the dictionaries for the catalog's snapshot (AnimeTools.load, at build time)."""
        with BUILD_LOCK, closing(sqlite3.connect(self.database, timeout=120)) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS entity_dictionary(kind TEXT,id TEXT,name TEXT,aliases TEXT,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS entity_keys(key TEXT,kind TEXT,id TEXT,PRIMARY KEY(key,kind,id));
                CREATE VIRTUAL TABLE IF NOT EXISTS entity_dictionary_fts USING fts5(
                    kind UNINDEXED,id UNINDEXED,names,tokenize='trigram');
            ''')
            stamp = db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone()[0] + ':dictionary-v1'
            if db.execute("SELECT value FROM meta WHERE key='entity_dictionary'").fetchone() == (stamp,):
                return
            db.execute('BEGIN IMMEDIATE')
            try:
                for table in ('entity_dictionary', 'entity_keys', 'entity_dictionary_fts'):
                    db.execute('DELETE FROM ' + table)
                def insert(kind, identifier, name, aliases):
                    aliases = list(dict.fromkeys([name, *aliases]))
                    db.execute('INSERT OR IGNORE INTO entity_dictionary VALUES (?,?,?,?)',
                               (kind, str(identifier), name, json.dumps(aliases, ensure_ascii=False)))
                    keys = {normalize(a) for a in aliases if a}
                    if kind == 'character':
                        keys |= {' '.join(sorted(key.split())) for key in list(keys)}
                    db.executemany('INSERT OR IGNORE INTO entity_keys VALUES (?,?,?)',
                                   [(key, kind, str(identifier)) for key in keys if key])
                    db.execute('INSERT INTO entity_dictionary_fts VALUES (?,?,?)',
                               (kind, str(identifier), ' | '.join(normalize(a) for a in aliases)))
                for aid, data in db.execute('SELECT id,data FROM anime').fetchall():
                    r = json.loads(data)
                    insert('anime', aid, r['title'], r['aliases'])
                for cid, name, aliases in db.execute("SELECT id,json_extract(data,'$.name'),json_extract(data,'$.aliases') FROM characters").fetchall():
                    insert('character', cid, name, json.loads(aliases))
                for kind, key in [('genre', 'genres'), ('tag', 'tags')]:
                    rows = db.execute("SELECT DISTINCT value FROM anime,json_each(json_extract(data,?))", ('$.' + key,)).fetchall()
                    for (name,) in rows:
                        insert(kind, normalize(name), name, [])
                db.execute("INSERT OR REPLACE INTO meta VALUES ('entity_dictionary',?)", (stamp,))
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def lookup(self, kind, query, limit=6, anime_ids=None):
        if kind not in KINDS:
            raise ValueError('kind must be anime, character, genre or tag')
        text = normalize(query)[:200]
        if not text:
            return {'status': 'empty_query', 'kind': kind, 'matches': []}
        keys = [text, ' '.join(sorted(text.split()))] if kind == 'character' else [text]
        scope = (' AND EXISTS (SELECT 1 FROM appearances p WHERE p.character_id=d.id AND p.anime_id IN (' +
                 ','.join('?' for _ in anime_ids) + '))') if kind == 'character' and anime_ids else ''
        scope_args = tuple(anime_ids or []) if scope else ()
        rows = self.catalog.query('SELECT DISTINCT d.id,d.name,d.aliases FROM entity_keys k JOIN entity_dictionary d '
                                  'ON d.kind=k.kind AND d.id=k.id WHERE k.kind=? AND k.key IN (' + ','.join('?' for _ in keys) + ')' + scope,
                                  (kind, *keys, *scope_args))
        exact = bool(rows)
        # Short mood words are not misspelled feature names: e.g. "sad"
        # must not resolve to "Sadism". Exact short names still work.
        if not exact and kind in ('genre', 'tag') and len(text) <= 3:
            return {'kind': kind, 'query': query, 'matches': [], 'status': 'no_matches'}
        if not rows:
            grams = list(dict.fromkeys(text[i:i+3] for i in range(len(text)-2) if text[i:i+3].strip()))[:80]
            if grams:
                expression = ' OR '.join('"' + gram.replace('"', '""') + '"' for gram in grams)
                rows = self.catalog.query('SELECT d.id,d.name,d.aliases FROM entity_dictionary_fts f '
                    'JOIN entity_dictionary d ON d.kind=f.kind AND d.id=f.id '
                    'WHERE entity_dictionary_fts MATCH ? AND f.kind=?' + scope + ' ORDER BY bm25(entity_dictionary_fts) LIMIT 160',
                    (expression, kind, *scope_args))
        if not rows:
            rows = self.catalog.query('SELECT d.id,d.name,d.aliases FROM entity_dictionary d WHERE kind=? AND name LIKE ?' + scope + ' LIMIT 80',
                                      (kind, query[:2] + '%', *scope_args))
        matches = []
        for identifier, name, raw_aliases in rows:
            aliases = json.loads(raw_aliases)
            score = max(max(SequenceMatcher(None, text, normalize(a)).ratio(),
                            SequenceMatcher(None, ' '.join(sorted(text.split())), ' '.join(sorted(normalize(a).split()))).ratio()) for a in aliases)
            for alias in aliases:
                normalized_alias = normalize(alias)
                if len(text) >= 3 and (text == normalized_alias or ' ' + text + ' ' in ' ' + normalized_alias + ' '):
                    score = max(score, .92)
                if len(text) >= 3 and any(ord(c) > 127 for c in text) and normalized_alias.startswith(text):
                    score = max(score, .92)
                # Compare a mistyped short title with the corresponding title
                # prefix, rather than lowering precision for the whole catalog.
                if kind in ('anime', 'character'):
                    prefix = ' '.join(normalized_alias.split()[:len(text.split())])
                    score = max(score, SequenceMatcher(None, text, prefix).ratio())
            threshold = .8 if kind == 'anime' else .65 if kind == 'character' else .85
            if not exact and score < threshold:
                continue
            matches.append({'kind': kind, 'id': identifier, 'name': name, 'aliases': aliases[:8],
                            'match': 'exact_or_name_order' if exact else 'fuzzy', 'similarity': round(score, 4)})
        matches.sort(key=lambda r: r['similarity'], reverse=True)
        return {'kind': kind, 'query': query, 'matches': matches[:limit], 'status': 'ok' if matches else 'no_matches'}

    def character_mentions(self, question):
        """Retrieve plausible full names, without selecting a question intent."""
        words = normalize(question).split()[:40]
        matches = {}
        for start in range(len(words)):
            for size in (2, 3, 4):
                phrase = words[start:start+size]
                if len(phrase) != size or any(w in STOP for w in phrase):
                    continue
                for match in self.lookup('character', ' '.join(phrase), 2)['matches']:
                    if match['similarity'] >= .9:
                        matches[match['id']] = match
        return sorted(matches.values(), key=lambda m: m['similarity'], reverse=True)[:5]

    def anime(self, identifier, section='overview', offset=0, limit=30):
        r = self.catalog.record(identifier)
        cast = self.catalog.query("SELECT c.id,json_extract(c.data,'$.name'),p.role FROM appearances p "
                                  "JOIN characters c ON c.id=p.character_id WHERE p.anime_id=? ORDER BY p.role='MAIN' DESC,c.id", (identifier,))
        if section == 'cast':
            page = cast[offset:offset+limit]
            # Short biographies belong in cast answers, without expanding the
            # ordinary overview/recommendation bundles or overflowing evidence.
            description_limit = min(600, 5000 // max(1, len(page)))
            profiles = dict(self.catalog.query(
                "SELECT c.id,substr(coalesce(json_extract(c.data,'$.description'),''),1,?) "
                "FROM appearances p JOIN characters c ON c.id=p.character_id "
                "WHERE p.anime_id=? ORDER BY p.role='MAIN' DESC,c.id LIMIT ? OFFSET ?",
                (description_limit, identifier, limit, offset)))
            return {'kind': 'anime', 'id': identifier, 'title': r['title'], 'source': r['source'],
                    'cast': [{'kind': 'character', 'id': cid, 'name': name, 'role': role,
                              'description': profiles.get(cid, '')} for cid, name, role in page],
                    'total_cast': len(cast), 'next_offset': offset + limit if offset + limit < len(cast) else None}
        if section in ('staff', 'studios'):
            rows = r.get(section, [])
            fields = ('name', 'role') if section == 'staff' else ('name', 'main', 'animation')
            return {'kind': 'anime', 'id': identifier, 'title': r['title'], 'source': r['source'],
                    'section': section,
                    section: [{key: person[key] for key in fields if key in person}
                              for person in rows[offset:offset + limit]],
                    'total_' + section: len(rows),
                    'next_offset': offset + limit if offset + limit < len(rows) else None}
        if section in ('relations', 'voices'):
            query = {'staff': 'staff director writer composer', 'studios': 'studio', 'relations': 'related sequel prequel', 'voices': 'voice'}[section]
            return {'kind': 'anime', 'id': identifier, 'title': r['title'], 'source': r['source'],
                    'section': section, 'data': self.catalog.detailed_fields(query, [r])}
        if section != 'overview':
            raise ValueError('Unknown anime section')
        main = [{'kind': 'character', 'id': cid, 'name': name, 'role': role} for cid, name, role in cast if role == 'MAIN']
        recommendations = self.catalog.query('SELECT a.id,r.votes,a.title,a.source FROM recommendations r '
                                             'JOIN anime a ON a.id=r.target_id WHERE r.anime_id=? ORDER BY r.votes DESC LIMIT 10', (identifier,))
        # Every scalar field and synopsis is included. Bulky relationships are
        # explicitly linked/paged, so missing profile text is never silent.
        description = r.get('description') or ''
        def recommendation(aid, votes):
            candidate = self.catalog.record(aid)
            return dict(self.brief(candidate), kind='anime', votes=votes,
                        shared_genres=sorted(set(r['genres']) & set(candidate['genres'])),
                        shared_tags=sorted(set(r['tags']) & set(candidate['tags'])))
        # Keep biographies in character profiles, rather than mixing every
        # supporting character into an anime/recommendation prompt.
        overview = {k: v for k, v in r.items() if k not in ('characters', 'recommendations', 'staff', 'search_data')}
        return {**overview, 'description': description[offset:offset+6000],
                'description_characters': len(description),
                'next_offset': offset+6000 if offset+6000 < len(description) else None,
                'kind': 'anime', 'main_characters': main, 'cast_count': len(cast),
                'recommendation_links': [{'kind': 'anime', 'id': aid, 'title': title, 'votes': votes, 'source': source}
                                         for aid, votes, title, source in recommendations],
                'recommendations': [recommendation(aid, votes)
                                    for aid, votes, _, _ in recommendations[:5]],
                'available_sections': ['overview', 'cast', 'staff', 'studios', 'relations', 'voices'],
                'character_profiles': 'Use get_character with a linked ID for its full profile.'}

    def character(self, identifier, offset=0, limit=8000, section='profile'):
        rows = self.catalog.query('SELECT data FROM characters WHERE id=?', (identifier,))
        if not rows:
            return {'status': 'not_found', 'kind': 'character', 'id': identifier}
        r = json.loads(rows[0][0])
        appearances = self.catalog.query('SELECT a.id,a.title,p.role,a.source FROM appearances p JOIN anime a ON a.id=p.anime_id '
                                         'WHERE p.character_id=? ORDER BY a.popularity DESC', (identifier,))
        description = r.get('description') or ''
        links = [{'kind': 'anime', 'id': aid, 'title': title, 'role': role, 'source': source}
                 for aid, title, role, source in appearances]
        if section == 'appearances':
            return {'kind': 'character', 'id': identifier, 'name': r['name'], 'appears_in': links[offset:offset+limit],
                    'appearance_count': len(links), 'next_offset': offset+limit if offset+limit < len(links) else None}
        return {**r, 'id': identifier, 'kind': 'character', 'description': description[offset:offset+limit],
                'description_characters': len(description), 'next_offset': offset+limit if offset+limit < len(description) else None,
                'appears_in': links[:12], 'appearance_count': len(links),
                'appearances_next_offset': 12 if len(links) > 12 else None}

    @staticmethod
    def anime_text(r):
        """The text a recommendation preference is compared with; `recommender build` embeds it for every anime."""
        return (r['title'] + '\n' + r.get('description', '')[:700] + '\n' +
                ', '.join(r.get('genres', []) + r.get('tags', [])))

    @staticmethod
    def brief(r):
        return {k: (r[k][:700] if k == 'description' else r[k]) for k in
                ('id', 'title', 'aliases', 'description', 'genres', 'tags', 'year', 'start_date', 'score', 'popularity',
                 'format', 'episodes', 'duration', 'status', 'adult', 'source_material', 'studios', 'source') if k in r}

    def recommendations(self, seed_ids, terms, limit=5, constraints=None):
        titles = [self.catalog.record(aid)['title'] for aid in seed_ids]
        records, reasons = self.catalog.recommend(' '.join(terms), limit, seed_titles=titles if titles else None,
                                                 **({'constraints': constraints} if constraints else {}))
        def brief(record):
            result = self.brief(record)
            if constraints and constraints.get('filters'):
                from .catalog_search import evidence_value
                result['matched_fields'] = {item['field']: evidence_value(
                                           record.get(item['field'], record.get('search_data', {}).get(item['field'])), item['value'])
                                           for item in constraints['filters'] if item['field'] != 'characters'}
            relations = record.get('search_data', {}).get('relations', [])
            if isinstance(relations, list):
                result['prior_related_works'] = [{'relation': item.get('relationType'),
                    'title': (item.get('node', {}).get('title') or {})}
                    for item in relations if item.get('relationType') in ('PREQUEL', 'PARENT')][:4]
            if constraints and (constraints.get('staff') or constraints.get('staff_role')):
                name = normalize(constraints.get('staff', ''))
                role = normalize(constraints.get('staff_role', ''))
                result['matched_staff'] = [{'name': person['name'], 'role': person['role']}
                    for person in record.get('staff', [])
                    if (not name or any(contains(key, name) or key == ' '.join(sorted(name.split()))
                                        for key in person['keys']))
                    and (not role or person['role_key'] == role)]
            return result
        return {'seed_anime': [{'id': aid, 'title': self.catalog.record(aid)['title']} for aid in seed_ids],
                'applied_filters': constraints or {},
                'items': [{**brief(r), 'reason': reasons.get(r['id'], ''), 'kind': 'anime'} for r in records]}
