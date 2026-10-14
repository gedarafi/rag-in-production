from __future__ import annotations

import requests

from .progress import Progress
from .tracing import tracer


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

    @tracer.llm
    def answer(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens,
                                "num_ctx": self.context_window},
                },
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.ollama_url}. Start Ollama and pull '{self.llm}'."
            ) from error
        return response.json()["response"].strip()

    def route_plan(self, prompt: str, *, schema=None) -> str:
        from .semantic_router import routing_schema
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": self.llm, "prompt": prompt, "stream": False, "format": schema or routing_schema(),
                "options": {"temperature": 0, "num_predict": 512, "num_ctx": self.context_window}, "keep_alive": "10m",
            }, timeout=90)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as error:
            raise RuntimeError(f"Could not reach Ollama for semantic routing at {self.ollama_url}: {error}") from error

    def plan_tools(self, prompt, schema):
        max_steps = schema['properties']['steps']['maxItems']
        try:
            response = requests.post(f'{self.ollama_url}/api/chat', json={
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': 'You are a tool planner. Produce the complete dependent function plan '
                     'in one JSON response. Do not write the user-facing answer. Use result references for IDs obtained '
                     'by earlier steps. Decide use_previous_context explicitly: true for follow-ups, false for independent '
                     'or new-topic questions. For preference recommendations, use semantic_feature_candidates directly '
                     'as genre/tag IDs; preferences are not anime titles. Read feature descriptions to distinguish meanings. '
                     'A self-contained request such as sad anime recommendations starts a new search even after '
                     'discussing a title: do not inherit that title as a seed. Only inherit subjects for an actual '
                     'reference such as "like that", an omitted factual subject, or a pending clarification. '
                     'Recommendation tools must use limit=5 unless the user explicitly requests a different count. '
                     'If supplied evidence already answers the question, steps should be empty.'},
                    {'role': 'user', 'content': prompt}], 'stream': False, 'format': schema,
                'options': {'temperature': 0, 'num_predict': min(4096, max(1024, max_steps * 200)),
                            'num_ctx': self.context_window}, 'keep_alive': '10m'}, timeout=180)
            response.raise_for_status()
            return response.json()['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for tool planning: {error}') from error

    def agent_step(self, prompt, schema):
        grounded = any(b['properties']['action']['const'] == 'answer' for b in schema['anyOf'])
        stage = ('Source evidence HAS ALREADY BEEN LOADED in tool_results. Read it first. '
                 'Use action=answer with a cited natural-language answer when it is sufficient. '
                 'Call another tool only for genuinely missing evidence. ' if grounded else
                 'No factual evidence has been loaded yet. Use action=call to retrieve it. ')
        try:
            response = requests.post(f'{self.ollama_url}/api/chat', json={
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
            response.raise_for_status()
            return response.json()['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the entity tool loop: {error}') from error

    def agent_reply(self, prompt, temperature, max_tokens):
        try:
            response = requests.post(f'{self.ollama_url}/api/chat', json={
                'model': self.llm, 'messages': [
                    {'role': 'system', 'content': 'Answer the person directly in natural language. Use only supplied '
                     'source evidence for facts. Never refer to the person as the user or narrate your internal process. '
                     'Do not output JSON or function calls. If evidence is missing, ask your own concise clarification.'},
                    {'role': 'user', 'content': prompt}], 'stream': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': self.context_window},
                'keep_alive': '10m'}, timeout=180)
            response.raise_for_status()
            return response.json()['message']['content']
        except requests.RequestException as error:
            raise RuntimeError(f'Could not reach Ollama for the final answer: {error}') from error
