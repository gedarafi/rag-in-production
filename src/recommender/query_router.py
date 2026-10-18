"""Validated plan types and deterministic execution of selected catalog tools."""
from __future__ import annotations

from dataclasses import dataclass
import json

from .anime_tools import contains, normalize


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    subject: str
    limit: int = 5
    field: str | None = None
    entity_type: str | None = None
    anime_scope: str | None = None
    reason: str | None = None
    seed_titles: tuple[str, ...] = ()


def tool_status(status, **data):
    """Machine-readable tool feedback; the answer model supplies all wording."""
    return json.dumps({'status': status, **data}, ensure_ascii=False)


def excerpt(description: str, maximum: int = 700) -> str:
    paragraphs = [p.strip() for p in description.split("\n\n") if p.strip()]
    narrative = "\n\n".join(p for p in paragraphs if not p.startswith(("__", "**"))) or description
    if len(narrative) <= maximum:
        return narrative
    end = narrative.rfind(". ", maximum // 2, maximum)
    return narrative[:end + 1] if end != -1 else narrative[:maximum].rsplit(" ", 1)[0] + "…"


def fast_answer(tools, question, history=None, *, plan: QueryPlan):
    if plan.intent == 'clarify':
        return tool_status('clarification_requested', subject=plan.subject, reason=plan.reason)
    if plan.intent == "cast":
        records = []
        subject = plan.subject
        if subject:
            records = tools.titles(subject, 1)
            # Match the extracted title, never incidental words in the question.
            if records and not any(normalize(subject) == normalize(a) or contains(normalize(a), normalize(subject))
                                   for a in records[0]['aliases']):
                records = []
            if not records:
                return tool_status('not_found', entity_type='anime', subject=subject)
        else:
            return tool_status('missing_subject', tool='cast')
        if not records:
            return tool_status('missing_subject', tool='cast')
        record = records[0]
        if hasattr(tools, 'record'):
            record = tools.record(record['id'], True)
        names = list(dict.fromkeys(c['name'] for c in record.get('characters', []) if c['role'] == 'MAIN'))
        if not names:
            return tool_status('missing_field', title=record['title'], field='main_characters', source=record['source'])
        profiles = '\n'.join(
            f"**{c['name']}**: {excerpt(c.get('description') or '') or 'Biography unavailable; documented role: main character.'}"
            for c in record.get('characters', []) if c['role'] == 'MAIN')
        return f"**{record['title']}** — main characters: {', '.join(names)}. [{record['source']}]\n\n{profiles}"
    if plan.intent == "detail":
        return None
    if plan.intent == "popular":
        return tools.listing(tools.popular(plan.limit, tools.filters(question)),
                             "Most popular in the imported AniList snapshot, sorted by the numeric popularity field.")
    if plan.intent == "count":
        if not hasattr(tools, "count"):
            return None
        return f"The imported AniList snapshot contains **{tools.count(tools.filters(question)):,}** matching anime. [catalog.jsonl]"
    if plan.intent == "recommend":
        effective = plan.subject
        if plan.seed_titles:
            records, reasons = tools.recommend(effective, plan.limit, seed_titles=plan.seed_titles)
            seeds = list(dict.fromkeys(r['title'] for title in plan.seed_titles for r in tools.titles(title, 1)))
            heading = 'Seed anime: ' + ', '.join(f'**{title}**' for title in seeds) + '. These are reference titles, not recommendations.\nRecommendations from the imported AniList snapshot.'
        else:
            records, reasons = tools.recommend(effective, plan.limit)
            heading = "Recommendations from the imported AniList snapshot."
        return tools.listing(records, heading, reasons)
    if plan.intent in ("character", "overview"):
        # "What is Naruto?" is the show; "Who is Naruto?" is the character.
        found = (tools.character(plan.subject, anime_scope=plan.anime_scope) if plan.anime_scope else
                 tools.character(plan.subject)) if plan.intent == "character" else None
        if found:
            char, record = found
            if not char['description']:
                return tool_status('missing_field', name=char['name'], field='description', source=record['source'])
            description = char['description']
            return f"**{char['name']}** — {char['role'].lower()} character in **{record['title']}**.\n\n{excerpt(description)}\n\n[{record['source']}]"
        if plan.intent == "character":
            return tool_status('not_found', entity_type='character', subject=plan.subject, anime_scope=plan.anime_scope)
        records = tools.titles(plan.subject, 1)
        if records:
            record = records[0]
            subject = normalize(plan.subject)
            if not any(subject == normalize(a) or contains(normalize(a), subject) for a in record["aliases"]):
                return tool_status('not_found', entity_type='anime', subject=plan.subject)
            if not record['description']:
                return tool_status('missing_field', title=record['title'], field='description', source=record['source'])
            description = record['description']
            return f"**{record['title']}**\nAliases: {', '.join(record['aliases'][:5])}\n\n{excerpt(description)}\n\nGenres: {', '.join(record['genres'])}. [{record['source']}]"
        return tool_status('not_found', entity_type='anime', subject=plan.subject)
    records = tools.titles(plan.subject, 1)
    if not records:
        return tool_status('not_found', entity_type='anime', subject=plan.subject, field=plan.field)
    record = records[0]
    if plan.field == "characters":
        if hasattr(tools, "record"):
            record = tools.record(record["id"], True)
        value = ", ".join(c["name"] for c in record.get("characters", []) if c["role"] == "MAIN")
    else:
        value = record.get(plan.field)
        if plan.field == 'duration' and value is not None:
            episodes = record.get('episodes')
            if episodes is not None:
                return (f"**{record['title']}** — {episodes:g} episodes, {value:g} minutes per episode; "
                        f"estimated total runtime {episodes * value:g} minutes (episode count × per-episode runtime; "
                        f"excluding extras). [{record['source']}]")
        if isinstance(value, list):
            value = ", ".join(value)
        elif isinstance(value, float):
            value = f"{value:g}"
    label = {"year": "Start year", "duration": "Minutes per episode", "score": "Average score (0–100)"}.get(plan.field, plan.field.capitalize())
    if value in (None, ''):
        return tool_status('missing_field', title=record['title'], field=plan.field, source=record['source'])
    return f"**{record['title']}** — {label}: {value}. [{record['source']}]"
