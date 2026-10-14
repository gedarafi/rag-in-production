"""Indexed canonical catalog and bounded lexical retrieval, independent of models."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from itertools import permutations
from contextlib import closing
from pathlib import Path

from .anime_tools import AnimeTools, contains, normalize, number

SCHEMA_VERSION = "3"
BUILD_LOCK = threading.Lock()
STOP = set("a an the is are was were what who how why when where which please can could would you me i "
           "tell give explain describe about in of on to and or for from it its this that he she him her his their they them anime show shows "
           "character characters detail details detailed more does do s with like similar recommend something".split())

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS anime(id TEXT PRIMARY KEY, source TEXT UNIQUE, title TEXT, popularity REAL,
 score REAL, year INTEGER, format TEXT, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS anime_popularity ON anime(popularity DESC, id);
CREATE INDEX IF NOT EXISTS anime_filters ON anime(year, format);
CREATE TABLE IF NOT EXISTS aliases(alias TEXT COLLATE NOCASE, anime_id TEXT,
 PRIMARY KEY(alias, anime_id));
CREATE TABLE IF NOT EXISTS features(anime_id TEXT, term TEXT, label TEXT, PRIMARY KEY(anime_id, term));
CREATE INDEX IF NOT EXISTS feature_lookup ON features(term, anime_id);
CREATE TABLE IF NOT EXISTS characters(id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS character_aliases(alias TEXT COLLATE NOCASE, character_id TEXT,
 PRIMARY KEY(alias, character_id));
CREATE TABLE IF NOT EXISTS appearances(anime_id TEXT, character_id TEXT, role TEXT,
 PRIMARY KEY(anime_id, character_id));
CREATE INDEX IF NOT EXISTS appearance_character ON appearances(character_id, anime_id);
CREATE TABLE IF NOT EXISTS recommendations(anime_id TEXT, target_id TEXT, votes REAL,
 PRIMARY KEY(anime_id, target_id));
CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(text, source UNINDEXED, anime_id UNINDEXED,
 entity_type UNINDEXED, character_id UNINDEXED, tokenize='unicode61');
"""


def grams(query: str) -> list[str]:
    words = normalize(query).split()[:60]
    return list({" ".join(words[i:j]) for i in range(len(words))
                 for j in range(i + 1, min(len(words), i + 14) + 1)
                 if j > i + 1 or words[i] not in STOP})


def ensure_database(folder: Path) -> Path:
    source = folder / "catalog.jsonl"
    database = folder / "catalog.sqlite3"
    stamp = f"{SCHEMA_VERSION}:{source.stat().st_mtime_ns}:{source.stat().st_size}"
    with BUILD_LOCK, closing(sqlite3.connect(database, timeout=120)) as db:
        db.executescript(SCHEMA)
        if db.execute("SELECT value FROM meta WHERE key='snapshot'").fetchone() == (stamp,):
            return database
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("BEGIN IMMEDIATE")
        try:
            for table in ("anime", "aliases", "features", "characters", "character_aliases",
                          "appearances", "recommendations", "passages"):
                db.execute(f"DELETE FROM {table}")
            with source.open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        _insert(db, json.loads(line), folder)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('snapshot', ?)", (stamp,))
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return database


def _insert(db, record, folder):
    # Upgrade older exports' scalar facts from their associated document once.
    if "episodes" not in record or record.get('tag_schema') != 2:
        document = folder / record["source"]
        if document.is_file():
            with document.open(encoding="utf-8") as file:
                for line in file:
                    key, _, value = line.partition(":")
                    if key in {"episodes", "duration", "status"}:
                        record[key] = value.strip() if key == "status" else number(value)
                    if key == "tags":
                        try:
                            tags = json.loads(value)
                            record['tags'] = [t['name'] for t in tags if isinstance(t, dict) and t.get('name')]
                            record['tag_schema'] = 2
                        except (ValueError, TypeError):
                            pass
                        break
    compact = {k: v for k, v in record.items() if k not in ("characters", "recommendations")}
    aid = record["id"]
    db.execute("INSERT INTO anime VALUES (?,?,?,?,?,?,?,?)", (aid, record["source"], record["title"],
               record["popularity"], record["score"], record["year"], record["format"], json.dumps(compact, ensure_ascii=False)))
    db.executemany("INSERT OR IGNORE INTO aliases VALUES (?,?)", [(normalize(a), aid) for a in record["aliases"]])
    db.executemany("INSERT OR IGNORE INTO features VALUES (?,?,?)",
                   [(aid, normalize(t), t) for t in record["genres"] + record["tags"]])
    header = f"Anime: {record['title']}. Aliases: {', '.join(record['aliases'])}."
    _passages(db, header, record["description"], record["source"], aid, "anime", "")
    for char in record["characters"]:
        cid = str(char.get("id") or "name:" + normalize(char["name"]))
        db.execute("""INSERT INTO characters VALUES (?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data
          WHERE length(json_extract(excluded.data,'$.description')) > length(json_extract(characters.data,'$.description'))""",
                   (cid, json.dumps(char, ensure_ascii=False)))
        db.executemany("INSERT OR IGNORE INTO character_aliases VALUES (?,?)", [(normalize(a), cid) for a in char["aliases"]])
        db.execute("INSERT OR IGNORE INTO appearances VALUES (?,?,?)", (aid, cid, char["role"]))
        title = f"{header}\nCharacter: {char['name']}. Names: {', '.join(char['aliases'][:5])}."
        _passages(db, title, char["description"], record["source"], aid, "character", cid)
    db.executemany("INSERT OR IGNORE INTO recommendations VALUES (?,?,?)",
                   [(aid, r["id"], r["votes"]) for r in record["recommendations"]])


def _passages(db, header, body, source, anime_id, entity_type, character_id):
    body = " ".join(body.split()) or "No description provided in the imported dataset."
    for start in range(0, len(body), 1050):
        db.execute("INSERT INTO passages VALUES (?,?,?,?,?)", (header + "\n" + body[start:start + 1200],
                   source, anime_id, entity_type, character_id))


class SqliteAnimeTools(AnimeTools):
    def __init__(self, database: Path):
        self.database = database
        self.folder = database.parent
        self.terms = {row[0]: row[1] for row in self.query("SELECT DISTINCT term,label FROM features")}

    def query(self, sql, parameters=()):
        # Connections belong to calls, so a GUI pipeline can cross worker threads.
        with closing(sqlite3.connect(self.database, timeout=120)) as db:
            return db.execute(sql, parameters).fetchall()

    def record(self, identifier: str, full: bool = False) -> dict:
        record = json.loads(self.query("SELECT data FROM anime WHERE id=?", (identifier,))[0][0])
        if full:
            record["characters"] = [dict(json.loads(data), role=role) for data, role in self.query(
                "SELECT c.data,p.role FROM appearances p JOIN characters c ON c.id=p.character_id WHERE p.anime_id=?", (identifier,))]
            record["recommendations"] = [{"id": target, "votes": votes} for target, votes in self.query(
                "SELECT target_id,votes FROM recommendations WHERE anime_id=?", (identifier,))]
        return record

    def record_by_source(self, source: str) -> dict | None:
        rows = self.query("SELECT id FROM anime WHERE source=?", (source,))
        return self.record(rows[0][0], True) if rows else None

    def titles(self, query: str, limit: int = 5) -> list[dict]:
        normalized = normalize(query)
        terms = grams(query)
        if not normalized or not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        rows = self.query(f"SELECT a.alias,a.anime_id,n.popularity FROM aliases a JOIN anime n ON n.id=a.anime_id "
                          f"WHERE a.alias IN ({placeholders}) OR a.alias LIKE ? LIMIT 300", (*terms, normalized + "%"))
        matches = {}
        for alias, aid, popularity in rows:
            exact, contained = alias == normalized, contains(normalized, alias)
            if not exact and (alias in STOP or len(alias) < 3):
                continue
            score = (exact, contained, len(alias) if contained else -len(alias), popularity or 0)
            matches[aid] = max(matches.get(aid, score), score)
        return [self.record(aid) for aid in sorted(matches, key=matches.get, reverse=True)[:limit]]

    def _where(self, filters):
        clauses, params = ["1=1"], []
        for term in filters.get("terms", []):
            clauses.append("EXISTS (SELECT 1 FROM features f WHERE f.anime_id=a.id AND f.term=?)")
            params.append(normalize(term))
        for term in filters.get("exclude_terms", []):
            clauses.append("NOT EXISTS (SELECT 1 FROM features f WHERE f.anime_id=a.id AND f.term=?)")
            params.append(normalize(term))
        for key in ("year", "format"):
            if filters.get(key) is not None:
                clauses.append(f"a.{key}=?")
                params.append(filters[key])
        return " AND ".join(clauses), params

    def popular(self, limit=5, filters=None):
        where, params = self._where(filters or {})
        return [json.loads(r[0]) for r in self.query(f"SELECT a.data FROM anime a WHERE {where} "
                "AND a.popularity IS NOT NULL ORDER BY a.popularity DESC,a.id LIMIT ?", (*params, limit))]

    def count(self, filters=None):
        where, params = self._where(filters or {})
        return self.query(f"SELECT COUNT(*) FROM anime a WHERE {where}", params)[0][0]

    def recommend(self, query, limit=5, seed_titles=None):
        normalized = normalize(query)
        match = re.search(r"\b(?:similar to|like|based on|enjoyed|loved|liked|finished)\s+(.+)", normalized)
        subject = re.split(r"\s+(?:with|but|from|released|what|recommend|suggest)\s+", match[1])[0] if match else normalized
        seeds = ([r for title in seed_titles for r in self.titles(title, 1)] if seed_titles is not None else
                 self.titles(subject) if match or not self.filters(query)["terms"] else [])
        seeds = list({r['id']: r for r in seeds}.values())
        if seeds and seed_titles is None:
            explicit = [r for r in seeds if any(contains(subject, normalize(a)) for a in r["aliases"])]
            seeds = explicit or seeds[:1]
        filter_query = normalized
        for seed in seeds:
            for alias in sorted(seed["aliases"], key=len, reverse=True):
                filter_query = filter_query.replace(normalize(alias), " ")
        filters = self.filters(filter_query)
        if not seeds:
            if not any(filters.get(k) for k in ("terms", "exclude_terms", "year", "format")):
                return [], {}
            rows = self.popular(limit, filters)
            return rows, {r["id"]: "Matches: " + ", ".join(filters["terms"]) for r in rows}
        ids = [r["id"] for r in seeds]
        marks = ",".join("?" for _ in ids)
        features = {row[0] for row in self.query(f"SELECT DISTINCT term FROM features WHERE anime_id IN ({marks})", ids)}
        where, params = self._where(filters)
        # Candidate scores and limits are computed in SQL; only winners are hydrated.
        sql = f"""WITH seed_features AS (SELECT DISTINCT term FROM features WHERE anime_id IN ({marks})),
          votes AS (SELECT target_id,SUM(votes) AS vote FROM recommendations WHERE anime_id IN ({marks}) GROUP BY target_id),
          shared AS (SELECT f.anime_id,COUNT(*) AS count FROM features f JOIN seed_features s ON s.term=f.term GROUP BY f.anime_id),
          totals AS (SELECT anime_id,COUNT(*) AS count FROM features GROUP BY anime_id)
          SELECT a.id,COALESCE(v.vote,0) FROM anime a LEFT JOIN votes v ON v.target_id=a.id
          LEFT JOIN shared s ON s.anime_id=a.id LEFT JOIN totals t ON t.anime_id=a.id
          WHERE {where} AND a.id NOT IN ({marks}) AND (COALESCE(v.vote,0)>0 OR COALESCE(s.count,0)>0)
          ORDER BY COALESCE(v.vote,0)>0 DESC,COALESCE(v.vote,0) DESC,
          CAST(COALESCE(s.count,0) AS REAL)/MAX(1,?+COALESCE(t.count,0)-COALESCE(s.count,0)) DESC,
          a.popularity DESC,a.id LIMIT ?"""
        winners = self.query(sql, (*ids, *ids, *params, *ids, len(features), limit))
        records, reasons = [], {}
        for aid, vote in winners:
            record = self.record(aid)
            labels = sorted(t for t in record["genres"] + record["tags"] if normalize(t) in features)
            records.append(record)
            reasons[aid] = (f"AniList recommendation votes: {vote:g}. " if vote > 0 else "") + "Shared genres/tags: " + ", ".join(labels[:6])
        return records, reasons

    def character(self, query, anime_scope=None):
        subject = normalize(query)
        subject = re.sub(r"^(?:who is|who s|describe|tell me about|character description (?:of|for))\s+", "", subject)
        scope = re.search(r"\s+(?:from|in)\s+(.+)$", subject)
        scope_title = anime_scope or (scope[1] if scope else None)
        scoped = self.titles(scope_title, 20) if scope_title else None
        if scoped is not None:
            # A short alias such as "Zero" is not evidence for the supplied
            # franchise "Re:Zero". Scope may never widen to unrelated titles.
            scope_name = normalize(scope_title)
            scoped = [r for r in scoped if any(normalize(a) == scope_name or contains(normalize(a), scope_name)
                                               for a in r['aliases'])]
        if scope:
            subject = subject[:scope.start()]
        terms = grams(subject)
        words = subject.split()
        if 2 <= len(words) <= 4:
            terms = list(set(terms) | {' '.join(parts) for parts in permutations(words)})
        if not terms:
            return None
        marks = ",".join("?" for _ in terms)
        sql = f"""SELECT c.data,p.anime_id,p.role,n.alias,a.popularity FROM character_aliases n
          JOIN characters c ON c.id=n.character_id JOIN appearances p ON p.character_id=c.id
          JOIN anime a ON a.id=p.anime_id WHERE (n.alias IN ({marks}) OR n.alias LIKE ?)"""
        params = [*terms, subject + "%"]
        if scoped is not None:
            if not scoped:
                return None
            sql += " AND p.anime_id IN (" + ",".join("?" for _ in scoped) + ")"
            params += [r["id"] for r in scoped]
        sql += " ORDER BY a.popularity DESC LIMIT 300"
        matches = self.query(sql, params)
        requested = set(words)
        # Every supplied name component must match. Popularity cannot turn an
        # unmatched full name into a one-token hit from another series.
        matches = [r for r in matches if requested.issubset(set(r[3].split()))]
        if not matches:
            return None
        row = max(matches, key=lambda r: (set(r[3].split()) == requested and len(words) > 1,
                                         r[4] or 0, r[3] == subject, r[2] == "MAIN"))
        return dict(json.loads(row[0]), role=row[2]), self.record(row[1])

    def lexical(self, query, limit=4, anime_ids=None, entity_type=None, character_id=None):
        tokens = [t for t in normalize(query).split() if t not in STOP][:16]
        conditions, params = [], []
        if anime_ids:
            conditions.append("anime_id IN (" + ",".join("?" for _ in anime_ids) + ")")
            params += list(anime_ids)
        if entity_type:
            conditions.append("entity_type=?")
            params.append(entity_type)
        if character_id is not None:
            conditions.append("character_id=?")
            params.append(str(character_id))
        suffix = " AND " + " AND ".join(conditions) if conditions else ""
        if not tokens:
            if not conditions:
                return []
            rows = self.query("SELECT text,source,anime_id,entity_type,character_id FROM passages WHERE " +
                              " AND ".join(conditions) + " ORDER BY rowid LIMIT ?", (*params, limit))
            return [dict(zip(("text", "source", "anime_id", "entity_type", "character_id"), row)) for row in rows]
        for operator in (" AND ", " OR "):
            expression = operator.join('"' + t + '"' for t in dict.fromkeys(tokens))
            rows = self.query("SELECT text,source,anime_id,entity_type,character_id FROM passages "
                              "WHERE passages MATCH ?" + suffix + " ORDER BY bm25(passages) LIMIT ?", (expression, *params, limit))
            if rows:
                return [dict(zip(("text", "source", "anime_id", "entity_type", "character_id"), row)) for row in rows]
        return []

    def detailed_fields(self, query, records):
        """Open original exports only for requested fields absent from summaries."""
        query = normalize(query)
        keys = set()
        if any(word in query for word in ("director", "directed", "staff", "writer", "composer")):
            keys.add("staff")
        if any(word in query for word in ("studio", "produced", "producer")):
            keys.add("studios")
        if any(word in query for word in ("sequel", "prequel", "adaptation", "related")):
            keys.add("relations")
        if any(word in query for word in ("voice", "voiced", "seiyuu")):
            keys.add("characters")
        if not keys:
            return []
        evidence = []
        for record in records[:2]:
            path = self.folder / record["source"]
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as file:
                for line in file:
                    key, _, value = line.partition(":")
                    if key not in keys:
                        continue
                    try:
                        items = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    facts = []
                    for item in items:
                        node = item.get("node") or item
                        if key == "staff":
                            role = item.get("role", "")
                            if "direct" in query and "director" not in role.casefold():
                                continue
                            facts.append({"name": node.get("name"), "role": role})
                        elif key == "studios":
                            facts.append({"name": node.get("name"), "main": item.get("isMain")})
                        elif key == "relations":
                            facts.append({"title": node.get("title"), "relation": item.get("relationType")})
                        else:
                            voices = [{"name": v.get("name"), "language": v.get("languageV2")} for v in item.get("voiceActors", [])]
                            name = node.get("name") or item.get("name")
                            if name and voices:
                                facts.append({"character": name, "voices": voices})
                    if facts:
                        evidence.append({"text": f"Anime: {record['title']}. {key}: " + json.dumps(facts[:20], ensure_ascii=False),
                                         "source": record["source"], "anime_id": record["id"], "entity_type": key})
        return evidence
