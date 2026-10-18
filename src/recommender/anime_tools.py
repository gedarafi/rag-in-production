"""Local, deterministic AniList tools. Numeric queries never use vector similarity."""
from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    # Fold Latin accents without erasing Japanese dakuten/handakuten.
    text = "".join("".join(d for d in unicodedata.normalize("NFKD", c)
                           if not unicodedata.combining(d))
                   if "LATIN" in unicodedata.name(c, "") else c for c in text)
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def contains(text: str, term: str) -> bool:
    return bool(term) and f" {term} " in f" {text} "


def prose(text: str | None) -> str:
    # AniList spoiler blocks are excluded from character descriptions by default.
    text = re.sub(r"~!.*?(?:!~|$)", "", text or "", flags=re.S)
    return html.unescape(re.sub(r"<[^>]*>", " ", text)).strip()


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def structured(row: dict, source: str) -> dict:
    """Extract useful fields without embedding images, staff JSON or URL lists."""
    def array(key):
        value = row.get(key) or []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        return value if isinstance(value, list) else []

    aliases = [row.get(k) for k in ("title_english", "title_romaji", "title_native", "title_userPreferred", "title")]
    aliases += array("synonyms")
    aliases = list(dict.fromkeys(prose(a) for a in aliases if isinstance(a, str) and a.strip()))
    characters = []
    for edge in array("characters"):
        node = edge.get("node") or edge
        name = node.get("name") or edge.get("name") or {}
        if isinstance(name, str):
            name = {"full": name}
        names = [name.get("full"), name.get("native"), *(name.get("alternative") or [])]
        names = [n for n in names if isinstance(n, str) and n.strip()]
        if names:
            characters.append({"id": node.get("id"), "name": names[0], "aliases": names,
                               "role": edge.get("role", ""), "description": prose(node.get("description"))})
    recommendations = []
    for edge in array("recommendations"):
        node = edge.get("node") or edge
        media = node.get("mediaRecommendation") or {}
        if media.get("id") is not None:
            recommendations.append({"id": str(media["id"]), "votes": number(node.get("rating")) or 0})
    from .catalog_metadata import production_metadata
    return {**production_metadata(row), "id": str(row.get("id", source)), "title": aliases[0] if aliases else source,
            "aliases": aliases, "description": prose(row.get("description")),
            "popularity": number(row.get("popularity")), "score": number(row.get("averageScore")),
            "year": number(row.get("startDate_year")), "format": row.get("format", ""),
            "episodes": number(row.get("episodes")), "duration": number(row.get("duration")),
            "status": row.get("status", ""),
            "adult": str(row.get("isAdult", "false")).lower() == "true",
            "genres": [g for g in array("genres") if isinstance(g, str)],
            "tag_schema": 2,
            "tags": [t["name"] for t in array("tags") if t.get("name")],
            "characters": characters, "recommendations": recommendations, "source": source}


class AnimeTools:
    def __init__(self, records: list[dict]):
        self.records = records
        self.by_id = {r["id"]: r for r in records}
        self.aliases = [(normalize(a), r) for r in records for a in r["aliases"] if a]
        self.terms = {normalize(t): t for r in records for t in r["genres"] + r["tags"]}

    @classmethod
    def load(cls, folder: Path) -> AnimeTools | None:
        """Build or refresh the SQLite catalog and its side tables from catalog.jsonl, then open it.

        This is the build path. A published bundle is opened as built instead (RagPipeline readonly).
        """
        path = folder / "catalog.jsonl"
        if not path.exists():
            return None
        from .catalog_metadata import ensure_production_metadata
        from .entity_store import EntityStore
        from .sqlite_catalog import BUILD_LOCK, SqliteAnimeTools, ensure_database
        import sqlite3
        try:
            database = ensure_database(folder)
            ensure_production_metadata(folder, database, BUILD_LOCK)
            tools = SqliteAnimeTools(database)
            EntityStore(tools).ensure_dictionary()
            return tools
        except sqlite3.Error as error:
            raise RuntimeError(f"Could not open or refresh the AniList SQLite catalog: {error}") from error

    def titles(self, query: str, limit: int = 5) -> list[dict]:
        query = normalize(query)
        matches = {}
        for alias, record in self.aliases:
            if alias != query and alias in {"what", "who", "anime", "popular", "with", "more", "something",
                                           "like", "similar", "another", "the", "show", "recommend", "me"}:
                continue
            if alias == query or contains(query, alias) or contains(alias, query):
                if len(alias) < 3 and alias != query:
                    continue
                score = (alias == query, contains(query, alias), len(alias) if contains(query, alias) else -len(alias),
                         record["popularity"] or 0)
                if record["id"] not in matches or score > matches[record["id"]][0]:
                    matches[record["id"]] = (score, record)
        return [r for _, r in sorted(matches.values(), key=lambda x: x[0], reverse=True)[:limit]]

    def filters(self, query: str) -> dict:
        query = normalize(query)
        excluded = [label for key, label in self.terms.items() if any(contains(query, prefix + " " + key)
                    for prefix in ("without", "not", "exclude", "excluding", "no"))]
        terms = [label for key, label in self.terms.items() if contains(query, key) and label not in excluded]
        # Prefer a complete tag ("Primarily Adult Cast") over a nested word ("Adult").
        terms = [t for t in terms if not any(normalize(t) != normalize(other) and
                 contains(normalize(other), normalize(t)) for other in terms)]
        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", query)]
        formats = {"movie": "MOVIE", "movies": "MOVIE", "tv": "TV", "ova": "OVA", "ona": "ONA"}
        return {"terms": terms, "exclude_terms": excluded, "year": years[-1] if years else None,
                "format": next((v for k, v in formats.items() if contains(query, k)), None)}

    def candidates(self, filters: dict) -> list[dict]:
        requested = {normalize(t) for t in filters.get("terms", [])}
        return [r for r in self.records if
                requested.issubset({normalize(t) for t in r["genres"] + r["tags"]}) and
                (filters.get("year") is None or r["year"] == filters["year"]) and
                (filters.get("format") is None or r["format"] == filters["format"])]

    def popular(self, limit: int = 5, filters: dict | None = None) -> list[dict]:
        records = [r for r in self.candidates(filters or {}) if r["popularity"] is not None]
        return sorted(records, key=lambda r: (-r["popularity"], r["id"]))[:limit]

    def recommend(self, query: str, limit: int = 5, seed_titles=None) -> tuple[list[dict], dict[str, str]]:
        normalized = normalize(query)
        subject = re.search(r"\b(?:similar to|like|based on)\s+(.+)", normalized)
        if subject:
            subject = re.split(r"\s+(?:with|but|from|released)\s+", subject[1])[0]
            seeds = self.titles(subject)
            if seeds and not any(normalize(a) == subject for a in seeds[0]["aliases"]):
                seeds = seeds[:1]
        elif self.filters(query)["terms"]:
            # Genre/tag requests must not pick up incidental titles such as
            # "With", "More", "Anime" or "Psychological" in the request text.
            seeds = []
        else:
            seeds = self.titles(query)
        if seeds:
            mentioned = subject if isinstance(subject, str) else normalized
            explicit = [r for r in seeds if any(contains(mentioned, normalize(a)) for a in r["aliases"])]
            seeds = explicit or seeds[:1]
        if seed_titles is not None:
            seeds = [r for title in seed_titles for r in self.titles(title, 1)]
        filters_query = normalize(query)
        for seed in seeds:
            for alias in sorted(seed["aliases"], key=len, reverse=True):
                filters_query = filters_query.replace(normalize(alias), " ")
        filters = self.filters(filters_query)
        if not seeds and not filters["terms"] and not filters["year"] and not filters["format"]:
            return [], {}
        seed_ids = {r["id"] for r in seeds}
        seed_features = {normalize(t) for r in seeds for t in r["genres"] + r["tags"]}
        votes = {}
        for seed in seeds:
            for rec in seed["recommendations"]:
                votes[rec["id"]] = votes.get(rec["id"], 0) + rec["votes"]
        ranked, reasons = [], {}
        for record in self.candidates(filters):
            if record["id"] in seed_ids:
                continue
            features = {normalize(t) for t in record["genres"] + record["tags"]}
            shared = seed_features & features
            vote = votes.get(record["id"], 0)
            if seeds and not shared and vote <= 0:
                continue
            similarity = len(shared) / max(1, len(seed_features | features))
            score = (vote > 0, vote, similarity, record["popularity"] or 0)
            ranked.append((score, record))
            if seeds:
                labels = sorted(t for t in record["genres"] + record["tags"] if normalize(t) in shared)
                reasons[record["id"]] = ((f"AniList recommendation votes: {vote:g}. " if vote > 0 else "") +
                                        "Shared genres/tags: " + ", ".join(labels[:6]))
            else:
                reasons[record["id"]] = "Matches: " + ", ".join(filters["terms"])
        return [r for _, r in sorted(ranked, key=lambda x: x[0], reverse=True)[:limit]], reasons

    def character(self, query: str) -> tuple[dict, dict] | None:
        query = normalize(query)
        query = re.sub(r"^(?:who is|who s|describe|tell me about|character description (?:of|for))\s+", "", query)
        scope = re.search(r"\s+(?:from|in)\s+(.+)$", query)
        scoped_ids = None
        if scope:
            scoped_ids = {r["id"] for r in self.titles(scope[1])}
            query = query[:scope.start()]
        matches = []
        for record in self.records:
            if scoped_ids is not None and record["id"] not in scoped_ids:
                continue
            for char in record["characters"]:
                scores = [(normalize(a) == query, contains(query, normalize(a)),
                           -len(normalize(a))) for a in char["aliases"]
                          if normalize(a) == query or contains(query, normalize(a)) or contains(normalize(a), query)]
                if scores:
                    score = max(scores)
                    # A short name is ambiguous: the character Naruto in a tiny
                    # unrelated show must not outrank Naruto Uzumaki merely
                    # because its full name is exactly one word.
                    rank = (score[0] and len(query.split()) > 1, record["popularity"] or 0,
                            score[0], char["role"] == "MAIN", score)
                    matches.append((rank, char, record))
        if not matches:
            return None
        _, char, record = max(matches, key=lambda x: x[0])
        return char, record

    @staticmethod
    def listing(records: list[dict], heading: str, reasons: dict | None = None) -> str:
        if not records:
            return json.dumps({'status': 'no_matches', 'records': []})
        lines = [heading, ""]
        for i, r in enumerate(records, 1):
            popularity = f"{r['popularity']:,.0f}" if r["popularity"] is not None else "unknown"
            line = f"{i}. **{r['title']}** — popularity: {popularity}; genres: {', '.join(r['genres'])}."
            if reasons and reasons.get(r["id"]):
                line += " " + reasons[r["id"]]
            if reasons is not None and r.get('description'):
                excerpt = ' '.join(r['description'].split())
                if len(excerpt) > 450:
                    excerpt = excerpt[:450].rsplit(' ', 1)[0] + '…'
                line += ' Synopsis excerpt: ' + excerpt
            lines.append(line + f" [{r['source']}]")
        return "\n".join(lines)

    def route(self, question: str, history: list[dict] | None = None, *, plan) -> str | None:
        from .query_router import fast_answer
        return fast_answer(self, question, history, plan=plan)
