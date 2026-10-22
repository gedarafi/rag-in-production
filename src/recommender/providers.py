"""The recommender's models: Ollama for every LLM step, sentence-transformers for embeddings.

Each method is one step of the pipeline with its own instructions; _ollama() turns it into an LLM
span named after the step (route_plan, plan_tools, agent_reply, ...), so a trace shows which step
asked what and what came back.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential
from openinference.instrumentation import get_input_attributes, get_llm_attributes, get_output_attributes

from .progress import Progress
from .errors import UserFacingError
from .prompt_library import PromptLibrary, library
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


# Throttling and the gateway errors a model service returns while it scales or restarts.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """How every LLM call survives throttling and transient failures (the [llm] settings)."""
    max_attempts: int = 4
    backoff_seconds: float = 1.0
    max_backoff_seconds: float = 20.0
    connect_timeout: float = 5.0

    @classmethod
    def from_settings(cls, settings) -> "RetryPolicy":
        return cls(settings.llm_max_attempts, settings.llm_backoff_seconds, settings.llm_max_backoff_seconds,
                   settings.llm_connect_timeout)


def transient(error: BaseException) -> bool:
    """A dropped or timed-out connection, or a status worth retrying; other HTTP errors are final."""
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return True
    return isinstance(error, requests.HTTPError) and getattr(error.response, "status_code", None) in TRANSIENT_STATUSES


def retry_after(error: BaseException | None) -> float | None:
    """Seconds the service asked to wait (Retry-After in seconds), if it did."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return max(0.0, float(headers.get("Retry-After")))
    except (TypeError, ValueError):
        return None


class LocalModels:
    render_recommendations = staticmethod(render_recommendations)

    def __init__(self, embedding_model: str, ollama_url: str, llm: str, device: str = "auto", batch_size: int = 64,
                 context_window: int = 8192, prompts: PromptLibrary | None = None,
                 retry: RetryPolicy | None = None) -> None:
        self._requested_device = device
        self._embedding_model = embedding_model
        self.batch_size = batch_size
        self._embedder = None
        self.ollama_url = ollama_url
        self.llm = llm
        self.context_window = context_window
        self.prompts = prompts or library()
        self.retry = retry or RetryPolicy()
        self.sleep = time.sleep

    def send(self, request, read_timeout: float):
        """request(timeout) with retries on transient failures; the last error is raised as it was.

        request makes one HTTP call and returns the response; it runs again on each attempt (so a
        signed request is signed afresh). Each retry is an `llm_retry` event on the current span.
        """
        policy = self.retry
        backoff = wait_random_exponential(multiplier=policy.backoff_seconds, max=policy.max_backoff_seconds)

        def wait(state):
            asked = retry_after(state.outcome.exception())
            return min(asked, policy.max_backoff_seconds) if asked is not None else backoff(state)

        def record(state):
            error = state.outcome.exception()
            event('llm_retry', attempt=state.attempt_number, error=f'{type(error).__name__}: {error}',
                  wait_seconds=round(state.next_action.sleep, 3))

        for attempt in Retrying(stop=stop_after_attempt(policy.max_attempts), wait=wait,
                                retry=retry_if_exception(transient), before_sleep=record,
                                sleep=lambda seconds: self.sleep(seconds), reraise=True):
            with attempt:
                response = request((policy.connect_timeout, read_timeout))
                response.raise_for_status()
        return response

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
            response = self.send(lambda limit: requests.post(f'{self.ollama_url}{path}', json=payload, timeout=limit),
                                 timeout)
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
                    {'role': 'system', 'content': self.prompts.render('plan_tools_system')},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': min(4096, max(1024, max_steps * 200)),
                            'num_ctx': self.context_window}, 'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for tool planning: {error}') from error

    def resolve_mentions(self, prompt, schema):
        return self._mention_json('resolve_mentions', prompt, schema, self.prompts.render('resolve_mentions_system'))

    def classify_mentions(self, prompt, schema):
        from .mention_interpreter import examples, instructions
        from difflib import SequenceMatcher
        payload, _, feedback = prompt.partition('\n')
        context = json.loads(payload)
        question = context['LAST_REQUEST']
        demonstrations = sorted(examples(self.prompts), key=lambda e: SequenceMatcher(None, question.casefold(), e['request'].casefold()).ratio())[-3:]
        request = question
        if context['history']:
            request = 'Prior conversation (data): ' + json.dumps(context['history'], ensure_ascii=False) + '\nCurrent request: ' + question
        if feedback:
            request += '\nValidation feedback for the current request: ' + feedback
        return self._mention_json('classify_mentions', request, schema, instructions(self.prompts), demonstrations)

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
                    {'role': 'system', 'content': self.prompts.render('interpret_request_system')},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': 1800, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not interpret the request: {error}') from error

    def agent_step(self, prompt, schema):
        grounded = any(b['properties']['action']['const'] == 'answer' for b in schema['anyOf'])
        try:
            body = self._ollama('agent_step', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': self.prompts.render('agent_step_system', grounded=grounded)},
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
                    {'role': 'system', 'content': self.prompts.render('agent_fact_reply_system', subjects=json.dumps(subjects),
                                                         measurements=json.dumps(measurements))},
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
            raise UserFacingError('The model returned an invalid grounded response. Please retry.') from error
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the grounded answer: {error}') from error

    def agent_reply(self, prompt, temperature, max_tokens):
        try:
            body = self._ollama('agent_reply', '/api/chat', {
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': self.prompts.render('agent_reply_system')},
                    {'role': 'user', 'content': prompt}], 'stream': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            return body['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the final answer: {error}') from error


class BedrockModels(LocalModels):
    """Every LLM step on Amazon Bedrock (bedrock-mantle Chat Completions); embeddings stay local.

    The pipeline steps above build Ollama payloads. _ollama() is the one place that sends them, so this
    class translates each payload to an OpenAI-compatible request instead: messages and options map
    one to one, and an Ollama `format` schema becomes `response_format: json_schema`. Not every
    Bedrock model enforces that schema, so a structured reply that isn't JSON is asked for once more.
    Errors are requests exceptions, as with Ollama, so the steps' error handling stays the same.
    """

    def __init__(self, embedding_model: str, region: str, llm: str, device: str = "auto", batch_size: int = 64,
                 context_window: int = 8192, timeout: float = 120, prompts: PromptLibrary | None = None,
                 retry: RetryPolicy | None = None) -> None:
        super().__init__(embedding_model, "", llm, device, batch_size, context_window, prompts, retry)
        self.region = region
        self.url = f"https://bedrock-mantle.{region}.api.aws/v1/chat/completions"
        self.timeout = timeout
        self._credentials = None

    def chat_completions(self, body: dict) -> dict:
        """One SigV4-signed call; boto3 has no bedrock-mantle client, so botocore signs and requests sends."""
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        if self._credentials is None:
            import botocore.session
            self._credentials = botocore.session.get_session().get_credentials()
            if self._credentials is None:
                raise requests.ConnectionError("No AWS credentials for Bedrock (instance role, profile or environment).")
        data = json.dumps(body).encode()

        def request(timeout):
            signed = AWSRequest(method="POST", url=self.url, data=data, headers={"content-type": "application/json"})
            SigV4Auth(self._credentials.get_frozen_credentials(), "bedrock-mantle", self.region).add_auth(signed)
            return requests.post(self.url, data=data, headers=dict(signed.headers.items()), timeout=timeout)

        return self.send(request, self.timeout).json()

    def _ollama(self, name: str, path: str, payload: dict, timeout: float) -> dict:
        messages = payload.get('messages') or [{'role': 'user', 'content': payload['prompt']}]
        options = payload['options']
        body = {'model': self.llm, 'messages': messages, 'temperature': options['temperature'],
                'max_tokens': options['num_predict']}
        if 'format' in payload:
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': name, 'schema': payload['format'], 'strict': True}}
        with tracer.start_as_current_span(name, openinference_span_kind='llm') as span:
            span.set_attributes({
                **get_llm_attributes(
                    provider='aws', system='openai', model_name=self.llm,
                    invocation_parameters={'max_tokens': body['max_tokens'], 'temperature': body['temperature'],
                                           'structured': 'format' in payload},
                    input_messages=messages),
                **get_input_attributes(messages[-1]['content']),
            })
            result = self.chat_completions(body)
            text = ((result.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
            if 'format' in payload:
                try:
                    json.loads(text)
                except ValueError as error:
                    event('structured_output_retry', error=str(error))
                    retry = {**body, 'messages': [*messages, {'role': 'assistant', 'content': text},
                                                  {'role': 'user', 'content': self.prompts.render('structured_output_retry', error=error)}]}
                    result = self.chat_completions(retry)
                    text = ((result.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
            usage = result.get('usage') or {}
            counts = {'prompt': usage.get('prompt_tokens'), 'completion': usage.get('completion_tokens'),
                      'total': usage.get('total_tokens')}
            counts = {key: count for key, count in counts.items() if isinstance(count, int)}
            span.set_attributes({
                **get_llm_attributes(output_messages=[{'role': 'assistant', 'content': text}],
                                     token_count=counts or None),
                **get_output_attributes(text),
            })
        # The shape the pipeline steps read: /api/chat's message and /api/generate's response.
        return {'message': {'content': text}, 'response': text}
