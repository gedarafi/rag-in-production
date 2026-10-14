"""Deterministic planner double for catalog/retrieval tests, not router tests."""
from dataclasses import replace

import re
from src.anime_tools import normalize
from src.query_router import QueryPlan


def echo_evidence(prompt, *args):
    """Tool/retrieval tests inspect evidence; prose generation has separate tests."""
    return prompt.split('Context:\n', 1)[1].split('\n\nQuestion:', 1)[0]


class CatalogRouterStub:
    def plan(self, question, history, tools):
        plan = plan_query(question)
        previous = [m['content'] for m in history if m['role'] == 'user']
        if plan.intent == 'cast' and not plan.subject:
            prior = plan_query(previous[-1]) if previous else None
            if prior and prior.intent in ('overview', 'character'):
                return replace(plan, subject=prior.subject)
            return QueryPlan('clarify', '')
        if plan.intent == 'fact':
            records = tools.titles(question, 1)
            if not records and previous:
                records = tools.titles(plan_query(previous[-1]).subject, 1)
            if records:
                return replace(plan, subject=records[0]['title'])
        if 'more' in question.lower() and previous:
            prior = plan_query(previous[-1])
            return QueryPlan('detail', prior.subject, entity_type='character' if prior.intent == 'character' else 'anime')
        if plan.intent == 'recommend' and previous and not tools.titles(question) and not tools.filters(question)['terms']:
            return replace(plan, subject=question + ' similar to ' + plan_query(previous[-1]).subject)
        if plan.intent == 'overview' and 'describe' in question.lower():
            return replace(plan, intent='character')
        if plan.intent == 'detail':
            return replace(plan, entity_type='character' if any(w in question.lower() for w in ('motivation', 'personality')) else 'anime')
        return plan


def plan_query(question: str) -> QueryPlan:
    query = normalize(question)
    query = re.sub(r"^(?:(?:can|could|would) you\s+|please\s+)+", "", query)
    query = re.sub(r"^tell me\s+(?=(?:who|what|how|when)\b)", "", query)
    query = re.sub(r"^who (.+) is$", r"who is \1", query)
    query = re.sub(r"^(?:give me|provide) (?:a |an )?(?:short |brief )?(?:description|summary|overview) (?:of|for)\s+", "summarize ", query)
    count = re.search(r"\b(?:top|recommend|suggest|give me|list|show me)\s+(\d+)\b", query)
    limit = min(20, max(1, int(count[1]))) if count else 5
    cast = re.search(r"\b(?:main characters?|main casts?|cast members|protagonists?|lead characters?)\b", query)
    if cast:
        tail = query[cast.end():].strip()
        subject = re.sub(r"^(?:of|in|from|for)\s+", "", tail)
        if not subject:
            possessive = re.search(r"(?:of\s+)?(.+?)\s+s\s*$", query[:cast.start()])
            if possessive:
                subject = re.sub(r"^(?:who are|who is|list|name|tell me about)\s+", "", possessive[1])
        if subject in {"it", "this", "that", "the anime", "this anime", "that anime", "the show"}:
            subject = ""
        return QueryPlan("cast", subject, limit, "characters")
    detailed = bool(re.search(r"\b(why|compare|comparison|relationship|motivations?|personality|analysis|analyze|detailed|depth)\b|in detail", query))
    if detailed:
        subject = re.sub(r"^(?:explain|describe|analyze|tell me about)\s+", "", query)
        subject = re.split(r"\s+(?:s\s+)?(?:personality|motivations?|relationship|in detail)\b", subject)[0]
        return QueryPlan("detail", subject, limit)
    if re.search(r"\b(recommend\w*|suggest\w*|similar|(?:anime|shows|something) like|watch next|watch after|what (?:else )?should i watch)\b", query):
        query = re.sub(r"\bwatch (?:next|after)\b", "similar to", query)
        return QueryPlan("recommend", query, limit)
    if re.search(r"\b(most popular|most watched|popular anime|popular shows|highest popularity|largest fanbase|biggest fanbase|most fans)\b", query):
        return QueryPlan("popular", query, limit)
    if re.search(r"\bhow many (?:anime|shows|movies|titles)\b", query):
        return QueryPlan("count", query, limit)
    fields = [
        ("episodes", r"\b(?:episodes?|episode count)\b"),
        ("duration", r"\b(?:duration|episode length|minutes per episode)\b"),
        ("year", r"\b(?:when (?:did|was|is)|what year|release year|start year|released|aired)\b"),
        ("popularity", r"\b(?:popularity|how popular)\b"),
        ("score", r"\b(?:score|rating|rated)\b"),
        ("genres", r"\bgenres?\b"), ("tags", r"\btags?\b"),
        ("format", r"\bformat\b"), ("status", r"\bstatus\b"),
    ]
    for field, pattern in fields:
        if re.search(pattern, query):
            return QueryPlan("fact", query, limit, field)
    if re.search(r"\b(?:main characters|main cast|cast members)\b", query):
        return QueryPlan("fact", query, limit, "characters")
    match = re.match(r"(?:who is|who s|who exactly is|character description (?:of|for))\s+(.+)", query)
    if match:
        return QueryPlan("character", match[1], limit)
    match = re.match(r"(?:what is|what s|what are|describe|tell me about|summarize|summary of|give me a summary of)\s+(.+)", query)
    if match:
        subject = re.sub(r"\s+(?:about|anime)$", "", match[1])
        return QueryPlan("overview", subject, limit)
    return QueryPlan("detail", query, limit)


