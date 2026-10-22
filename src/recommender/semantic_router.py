"""Retrieve tool examples, then generate and validate a constrained routing plan."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .anime_tools import contains, normalize
from .embedding_cache import cached_embed
from .prompt_library import prompts_for
from .query_router import QueryPlan

FIELDS = {"episodes", "duration", "year", "score", "popularity", "genres", "tags", "format", "status"}
INTENTS = {"cast", "character", "overview", "popular", "recommend", "count", "fact", "detail", "clarify"}
ROLE_WORDS = {'the', 'a', 'an', 'main', 'lead', 'character', 'characters', 'cast', 'casts',
              'protagonist', 'protagonists', 'person', 'it', 'him', 'her'}


class RoutingPlanError(RuntimeError):
    """Internal planning failure; chat can continue through the entity agent."""


def routing_schema(seed_options=None):
    schema = {'type': 'object', 'additionalProperties': False,
            'required': ['intent', 'subject', 'field', 'limit', 'confidence', 'entity_type', 'anime_scope', 'reason', 'seed_titles'],
            'properties': {
                'intent': {'type': 'string', 'enum': sorted(INTENTS)},
                'subject': {'type': 'string', 'maxLength': 2000},
                'field': {'enum': [None, *sorted(FIELDS)]},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'entity_type': {'enum': [None, 'anime', 'character']},
                'anime_scope': {'type': ['string', 'null']},
                'reason': {'type': ['string', 'null'], 'maxLength': 300},
                'seed_titles': {'type': 'array', 'maxItems': 5, 'uniqueItems': True,
                                'items': {'type': 'string', 'maxLength': 300}}}}
    if seed_options is not None:
        if seed_options:
            schema['properties']['seed_titles']['items']['enum'] = list(dict.fromkeys(seed_options))
        else:
            schema['properties']['seed_titles']['maxItems'] = 0
    return schema


class SemanticRouter:
    def __init__(self, settings, models):
        self.settings, self.models = settings, models
        # Next to the configured prompt (a published bundle ships prompts/), else the repository's.
        path = settings.prompt.with_name("routes.json")
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / "prompts/routes.json"
        self.examples = json.loads(path.read_text(encoding="utf-8"))
        self.vectors = None
        self.cache = OrderedDict()
        self.last_decision = {}

    def retrieve(self, question, titles, history=None):
        texts, entries = [], []
        for entry in self.examples:
            for example in entry["examples"]:
                texts.append(entry["description"] + "\nExample question: " + example)
                entries.append(entry)
        if self.vectors is None:
            fingerprint = hashlib.sha256(json.dumps(self.examples, sort_keys=True).encode()).hexdigest()
            namespace = json.dumps(["routing-v1", fingerprint, self.settings.embedding,
                                    self.settings.embedding_cache_version, getattr(self.models, 'cache_revision', None)])
            self.vectors = cached_embed(texts, self.models, self.settings.index / "routing.sqlite3", namespace)
        masked = normalize(question)
        for title in titles:
            for alias in sorted(title['aliases'], key=len, reverse=True):
                alias = normalize(alias)
                if len(alias) >= 3 and contains(masked, alias):
                    masked = (" " + masked + " ").replace(" " + alias + " ", " ENTITY ").strip()
        previous = [m['content'] for m in history or [] if m['role'] == 'user']
        if previous:
            masked += '\nPrevious user question: ' + previous[-1][:400]
        scores = self.models.embed([masked])[0] @ self.vectors.T
        retrieved, seen = [], set()
        for index in np.argsort(scores)[::-1]:
            entry = entries[index]
            key = (entry['intent'], entry.get('field'))
            if key not in seen:
                seen.add(key)
                retrieved.append(dict(entry, similarity=round(float(scores[index]), 4)))
            if len(retrieved) == self.settings.router_top_k:
                break
        return retrieved

    def plan(self, question, history, tools):
        cache_key = json.dumps([question, history], sort_keys=True, ensure_ascii=False)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            plan, decision = self.cache[cache_key]
            self.last_decision = dict(decision, cached=True)
            return plan
        titles = tools.titles(question, 5)
        current_titles = [{'title': r['title'], 'matched_aliases': [a for a in r['aliases'] if contains(normalize(question), normalize(a))]}
                          for r in titles]
        for message in history:
            if message['role'] == 'user':
                titles.extend(tools.titles(message['content'], 3))
        retrieved = self.retrieve(question, titles, history)
        candidates = [{"title": r['title'], "aliases": r['aliases'][:5]} for r in {r['id']: r for r in titles}.values()]
        from .sqlite_catalog import STOP
        for text in [question, *(m['content'] for m in history if m['role'] == 'user')]:
            if not set(normalize(text).split()) - STOP - ROLE_WORDS:
                continue
            found = tools.character(text)
            if found:
                char, record = found
                candidates.append({'title': char['name'], 'aliases': char['aliases'],
                                   'entity_type': 'character', 'anime': record['title']})
        prompts = prompts_for(self.settings)
        prompt = prompts.render('route_plan', data=json.dumps({
            "available_intents": sorted(INTENTS), "available_fact_fields": sorted(FIELDS),
            "retrieved_tools": retrieved, "known_entity_candidates": candidates,
            'current_title_candidates': current_titles,
            "history": history, "question": question}, ensure_ascii=False))
        feedback = []
        for attempt in range(3):
            correction = prompts.render('route_plan_correction', feedback=json.dumps(feedback, ensure_ascii=False),
                                        has_feedback=bool(feedback))
            seed_options = [r['title'] for r in candidates if r.get('entity_type') != 'character']
            raw = self.models.route_plan(prompt + correction, schema=routing_schema(seed_options))
            try:
                plan, confidence = self.validate(raw, question, history, candidates, retrieved)
                break
            except (ValueError, TypeError, KeyError) as error:
                feedback.append({'plan': raw, 'validation_error': str(error)})
        else:
            self.last_decision = {'status': 'invalid_plan', 'validation_feedback': feedback}
            raise RoutingPlanError('The routing model did not produce an executable plan after three validation attempts.')
        decision = {"intent": plan.intent, "confidence": confidence, 'validation_retries': attempt,
                    'subject': plan.subject, 'seed_titles': plan.seed_titles, 'anime_scope': plan.anime_scope,
                    "retrieved": [(r['intent'], r.get('field'), r['similarity']) for r in retrieved], "cached": False}
        self.last_decision = decision
        self.cache[cache_key] = (plan, decision)
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return plan

    def validate(self, raw, question, history, candidates, retrieved):
        data = json.loads(raw)
        intent, subject = data['intent'], data['subject']
        confidence, limit = data['confidence'], data.get('limit', 5)
        field = data.get('field')
        entity_type = data.get('entity_type')
        anime_scope = data.get('anime_scope')
        if anime_scope is not None and (not isinstance(anime_scope, str) or not anime_scope.strip() or len(anime_scope) > 500):
            raise ValueError('Invalid anime scope')
        if entity_type not in (None, 'anime', 'character'):
            raise ValueError('Invalid entity type')
        if intent not in INTENTS or not isinstance(subject, str) or len(subject) > 2000:
            raise ValueError('Invalid routing tool or subject')
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError('Invalid confidence')
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('Invalid result limit')
        if intent == 'fact' and field not in FIELDS:
            raise ValueError('Invalid fact field')
        reason = data.get('reason')
        seeds = data.get('seed_titles', [])
        if not isinstance(seeds, list) or len(seeds) > 5 or any(not isinstance(s, str) or not s.strip() or len(s) > 300 for s in seeds):
            raise ValueError('Invalid recommendation seed titles')
        if reason is not None and (not isinstance(reason, str) or len(reason) > 2000):
            raise ValueError('Invalid routing reason')
        if intent in {'cast', 'overview', 'character', 'fact'} or intent == 'detail' and entity_type == 'character':
            if subject.strip() and set(normalize(subject).split()).issubset(ROLE_WORDS):
                raise ValueError('A role or pronoun is not an identified subject')
            evidence = normalize(' '.join([question, *(m['content'] for m in history if m['role'] == 'user')]))
            aliases = {normalize(a) for r in candidates for a in [r['title'], *r['aliases']]}
            if subject.strip() and not contains(evidence, normalize(subject)) and normalize(subject) not in aliases:
                raise ValueError('Subject is unsupported by the question and history')
        if anime_scope:
            evidence = normalize(' '.join([question, *(m['content'] for m in history if m['role'] == 'user')]))
            aliases = {normalize(a) for r in candidates if r.get('entity_type') != 'character' for a in [r['title'], *r['aliases']]}
            # A full-name catalog match may supply its documented anime even
            # when the user only supplied the person's name. Assistant prose
            # still cannot authorize either entity or association.
            associated = [normalize(r['anime']) for r in candidates if r.get('entity_type') == 'character'
                          and r.get('anime') and set(normalize(subject).split()) == set(normalize(r['title']).split())]
            catalog_bound = any(name == normalize(anime_scope) or contains(name, normalize(anime_scope)) for name in associated)
            if not contains(evidence, normalize(anime_scope)) and normalize(anime_scope) not in aliases and not catalog_bound:
                raise ValueError('Anime scope is unsupported by user messages')
        if seeds:
            evidence = normalize(' '.join([question, *(m['content'] for m in history if m['role'] == 'user')]))
            aliases = {normalize(a) for r in candidates if r.get('entity_type') != 'character' for a in [r['title'], *r['aliases']]}
            for seed in seeds:
                if not contains(evidence, normalize(seed)) and normalize(seed) not in aliases:
                    raise ValueError('Recommendation seed is unsupported by user messages or catalog aliases')
        return QueryPlan(intent, subject, limit, field if intent == 'fact' else None, entity_type, anime_scope, reason,
                         tuple(dict.fromkeys(seeds))), confidence
