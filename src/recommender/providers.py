"""The recommender's models: Ollama for every LLM step, sentence-transformers for embeddings.

Each method is one step of the pipeline with its own instructions; _ollama() turns it into an LLM
span named after the step (route_plan, plan_tools, agent_reply, ...), so a trace shows which step
asked what and what came back.
"""
from __future__ import annotations

import json

import requests
from openinference.instrumentation import get_input_attributes, get_llm_attributes, get_output_attributes

from .progress import Progress
from .recommendation_output import render_recommendations
from ..tracing import annotate, event, tracer


def select_device(requested: str) -> str:
    import torch
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()}
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if not available.get(requested, False):
        raise ValueError(f"Embedding device '{requested}' is unavailable. Use auto or cpu, or install a compatible PyTorch build.")
    return requested


class LocalModels:
    render_recommendations = staticmethod(render_recommendations)

    def __init__(self, embedding_model: str, ollama_url: str, llm: str, device: str = "auto", batch_size: int = 64,
                 context_window: int = 8192) -> None:
        self._requested_device = device
        self._embedding_model = embedding_model
        self.batch_size = batch_size
        self._embedder = None
        self.ollama_url = ollama_url
        self.llm = llm
        self.context_window = context_window

    @property
    def device(self):
        return select_device(self._requested_device)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._embedder = SentenceTransformer(self._embedding_model, device=self.device, local_files_only=True)
            except OSError:
                self._embedder = SentenceTransformer(self._embedding_model, device=self.device)
        return self._embedder

    @embedder.setter
    def embedder(self, value):
        self._embedder = value

    @property
    def cache_revision(self):
        first = self.embedder._first_module()
        config = getattr(getattr(first, "auto_model", None), "config", None)
        return getattr(config, "_commit_hash", None)

    def embed(self, texts: list[str], progress: Progress | None = None):
        if progress:
            import numpy as np
            batch_size = 256
            vectors = None
            progress("Embedding chunks", 0, len(texts))
            for start in range(0, len(texts), batch_size):
                batch = self.embedder.encode(texts[start:start + batch_size], normalize_embeddings=True,
                                             batch_size=getattr(self, "batch_size", 64),
                                             show_progress_bar=False)
                if vectors is None:
                    vectors = np.empty((len(texts), batch.shape[1]), dtype=batch.dtype)
                vectors[start:start + len(batch)] = batch
                progress("Embedding chunks", start + len(batch), len(texts))
            return vectors
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                    batch_size=getattr(self, "batch_size", 64))

    def _ollama(self, name: str, path: str, payload: dict, timeout: float) -> dict:
        """One Ollama call, traced as an LLM span named after its pipeline step; the response body."""
        messages = payload.get('messages') or [{'role': 'user', 'content': payload['prompt']}]
        options = payload['options']
        with tracer.start_as_current_span(name, openinference_span_kind='llm') as span:
            span.set_attributes({
                **get_llm_attributes(
                    provider='ollama', system='ollama', model_name=self.llm,
                    invocation_parameters={'max_tokens': options['num_predict'], 'temperature': options['temperature'],
                                           'num_ctx': options['num_ctx'], 'structured': 'format' in payload},
                    input_messages=messages),
                **get_input_attributes(messages[-1]['content']),
            })
            response = requests.post(f'{self.ollama_url}{path}', json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            text = body['message']['content'] if 'message' in body else body['response']
            counts = {'prompt': body.get('prompt_eval_count'), 'completion': body.get('eval_count')}
            counts = {key: count for key, count in counts.items() if isinstance(count, int)}
            if len(counts) == 2:
                counts['total'] = counts['prompt'] + counts['completion']
            span.set_attributes({
                **get_llm_attributes(output_messages=[{'role': 'assistant', 'content': text}],
                                     token_count=counts or None),
                **get_output_attributes(text),
            })
            return body

    def answer(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            body = self._ollama(
                "answer", "/api/generate",
                {
                    "model": self.llm,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens,
                                "num_ctx": self.context_window},
                },
                timeout=300,
            )
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.ollama_url}. Start Ollama and pull '{self.llm}'."
            ) from error
        return body["response"].strip()

    def route_plan(self, prompt: str, *, schema=None) -> str:
        from .semantic_router import routing_schema
        try:
            body = self._ollama("route_plan", "/api/generate", {
                "model": self.llm, "prompt": prompt, "stream": False, "format": schema or routing_schema(),
                "options": {"temperature": 0, "num_predict": 512, "num_ctx": self.context_window}, "keep_alive": "10m",
            }, timeout=90)
            return body["response"]
        except requests.RequestException as error:
            raise RuntimeError(f"Could not reach Ollama for semantic routing at {self.ollama_url}: {error}") from error

    def plan_tools(self, prompt, schema):
        max_steps = schema['anyOf'][1]['properties']['steps']['maxItems']
        try:
            body = self._ollama('plan_tools', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': 'Plan read-only tools for the actual current anime-library request. '
                     'Return only the JSON plan. In-domain and uncertain requests have an empty direct_response. '
                     'Only clearly unrelated standalone tasks use out_of_domain, steps=[], use_previous_context=false '
                     'and a helpful direct_response. Unknown titles or missing anime keywords do not imply out_of_domain. '
                     'A follow-up with an omitted subject uses trusted prior subjects as a group: use_previous_context=true. '
                     'Do not confuse a new property question with a new subject. New independent requests do not inherit '
                     'old seeds or filters. Use supplied IDs and source evidence; request only genuinely missing data.'},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': min(4096, max(1024, max_steps * 200)),
                            'num_ctx': self.context_window}, 'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for tool planning: {error}') from error

    def resolve_mentions(self, prompt, schema):
        return self._mention_json('resolve_mentions', prompt, schema, 'Select catalog identities for previously classified mentions. '
                                  'Use the full request to distinguish adaptations and aliases. Return only schema-compliant JSON.')

    def classify_mentions(self, prompt, schema):
        from .mention_interpreter import examples, instructions
        from difflib import SequenceMatcher
        payload, _, feedback = prompt.partition('\n')
        context = json.loads(payload)
        question = context['LAST_REQUEST']
        demonstrations = sorted(examples(), key=lambda e: SequenceMatcher(None, question.casefold(), e['request'].casefold()).ratio())[-3:]
        request = question
        if context['history']:
            request = 'Prior conversation (data): ' + json.dumps(context['history'], ensure_ascii=False) + '\nCurrent request: ' + question
        if feedback:
            request += '\nValidation feedback for the current request: ' + feedback
        return self._mention_json('classify_mentions', request, schema, instructions(), demonstrations)

    def _mention_json(self, name, prompt, schema, system, examples=()):
        messages = [{'role': 'system', 'content': system}]
        for example in examples:
            messages.extend([{'role': 'user', 'content': example['request']},
                             {'role': 'assistant', 'content': json.dumps(example['output'], ensure_ascii=False)}])
        messages.append({'role': 'user', 'content': prompt})
        try:
            body = self._ollama(name, '/api/chat', {
                'model': self.llm, 'messages': messages,
                'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': 1800, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not resolve contextual mentions: {error}') from error

    def interpret_request(self, prompt, schema):
        try:
            body = self._ollama('interpret_request', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': 'Interpret user requirements using only the provided catalog schema. '
                     'Preserve positive requirements, negation scope and subjective preferences separately. Return JSON.'},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': 1800, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not interpret the request: {error}') from error

    def agent_step(self, prompt, schema):
        grounded = any(b['properties']['action']['const'] == 'answer' for b in schema['anyOf'])
        stage = ('Source evidence HAS ALREADY BEEN LOADED in tool_results. Read it first. '
                 'Use action=answer with a cited natural-language answer when it is sufficient. '
                 'Call another tool only for genuinely missing evidence. ' if grounded else
                 'No factual evidence has been loaded yet. Use action=call to retrieve it. ')
        try:
            body = self._ollama('agent_step', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': stage + 'You control an evidence-grounded assistant. '
                     'Return only the JSON decision. action=call is ONLY for requesting another tool execution, '
                     'not for presenting an answer. action=answer has tool=null and arguments={}. '
                     'For follow-ups, reference_anime identifies what the user means by that. '
                     'Recommended titles do not make this reference ambiguous. '
                     'A clarification must ask for missing information, never answer from memory.'},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': 900, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the entity tool loop: {error}') from error

    def agent_fact_reply(self, prompt, temperature, max_tokens, subjects, measurements=None, require_quantities=False):
        """One grounded generation; source units and generic arithmetic stay exact."""
        self.last_fact_validation_error = ''
        measurements = measurements or {}
        facts = {'type': 'array', 'minItems': len(subjects), 'maxItems': max(12, len(subjects)) if subjects else 0,
                 'items': {'type': 'object', 'additionalProperties': False, 'required': ['subject', 'fact'],
                           'properties': {'subject': {'enum': subjects} if subjects else {'type': 'string'},
                                          'fact': {'type': 'string', 'maxLength': 500}}}}
        prose = {'type': 'object', 'additionalProperties': False,
                 'required': ['kind', 'supporting_facts', 'answer'], 'properties': {
                     'kind': {'const': 'prose'}, 'supporting_facts': facts,
                     'answer': {'type': 'string', 'minLength': 1, **({'pattern': r'.*\?.*'} if not subjects else {})}}}
        quantity = {'type': 'object', 'additionalProperties': False,
                    'required': ['kind', 'supporting_facts', 'items'], 'properties': {
                        'kind': {'const': 'quantities'}, 'supporting_facts': facts,
                        'items': {'type': 'array', 'minItems': 1, 'maxItems': 12, 'items': {
                            'type': 'object', 'additionalProperties': False,
                            'required': ['label', 'expression', 'unit', 'decimals'], 'properties': {
                                'label': {'type': 'string', 'maxLength': 300},
                                'expression': {'type': 'string', 'maxLength': 500, 'pattern': r'^[0-9qc+*/().\s-]+$'},
                                'unit': {'enum': ['auto', 'minutes', 'hours', 'days', 'seconds']},
                                'decimals': {'type': 'integer', 'minimum': 0, 'maximum': 6}}}}}}
        schema = quantity if measurements and require_quantities else {'anyOf': [quantity, prose]} if measurements else prose
        try:
            body = self._ollama('agent_fact_reply', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': 'Return JSON. Select supporting facts for EACH subject from source records. '
                     'For numerical questions choose kind=quantities. Each item has an LLM-written label and an arithmetic '
                     'expression; the app evaluates it and appends the source-derived units. Use measurement variables '
                     'q0, q1, etc, or matching source numbers, with + - * / and parentheses. c0, c1 refer to earlier '
                     'items. Use unit=auto for counts or per-item measures; a time unit converts elapsed time. '
                     'Do NOT manually divide or multiply to convert time units: set unit and the app converts it. '
                     'The label names the subject/result without a numerical value or units. Choose the needed inputs '
                     'and operation yourself. Include per-subject facts before any requested combined result. '
                     'For additive quantities across multiple requested subjects, include a combined result after the '
                     'individual results. Do not combine non-additive facts. Expressions contain math ONLY, not unit words. '
                     'For explanations/comparisons use kind=prose and a concise natural answer. Previous assistant '
                     'replies are not source facts. If there are no source subjects, ask a question for the missing '
                     'information, never make a factual claim. Subjects: ' + json.dumps(subjects) +
                     '. Source measurements: ' + json.dumps(measurements)},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': max_tokens + 600, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            result = json.loads(body['message']['content'])
            if not subjects and result['supporting_facts']:
                raise ValueError('Cannot cite facts without source records')
            if subjects and set(subjects) != {row['subject'] for row in result['supporting_facts']}:
                raise ValueError('Missing source facts for a requested subject')
            if result['kind'] == 'quantities':
                if not measurements:
                    raise ValueError('No source measurements available')
                from .answer_arithmetic import render_quantities
                answer, calculations = render_quantities(result['items'], measurements)
            else:
                answer, calculations = result['answer'], []
                if not isinstance(answer, str) or not answer.strip() or (not subjects and '?' not in answer):
                    raise ValueError('Missing grounded answer or clarification')
            self.last_fact_grounding = {'supporting_facts': result['supporting_facts'], 'calculations': calculations}
            annotate(answer_grounding=self.last_fact_grounding)
            return answer
        except (ValueError, KeyError, TypeError) as error:
            self.last_fact_validation_error = str(error)
            event('grounding_validation_failed', error=str(error))
            raise RuntimeError('The model returned an invalid grounded response. Please retry.') from error
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the grounded answer: {error}') from error

    def agent_reply(self, prompt, temperature, max_tokens):
        try:
            body = self._ollama('agent_reply', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': 'Answer the person directly in natural language. Use only supplied '
                     'source evidence for facts. Never refer to the person as the user or narrate your internal process. '
                     'Do not output JSON or function calls. If evidence is missing, ask your own concise clarification.'},
                    {'role': 'user', 'content': prompt}], 'stream': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the final answer: {error}') from error
