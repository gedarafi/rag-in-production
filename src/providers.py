from __future__ import annotations

import json

import requests
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from sentence_transformers import SentenceTransformer

from .tracing import tracer


class LocalModels:
    def __init__(self, embedding_model: str, ollama_url: str, llm: str) -> None:
        self.embedder = SentenceTransformer(embedding_model)
        self.ollama_url = ollama_url
        self.llm = llm

    def embed(self, texts: list[str]):
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def answer(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        template: tuple[str, dict[str, str]] | None = None,
    ) -> str:
        options = {"temperature": temperature, "num_predict": max_tokens}
        with tracer.start_as_current_span(
            "ollama.generate",
            attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value},
        ) as span:
            span.set_attributes({
                SpanAttributes.LLM_PROVIDER: "ollama",
                SpanAttributes.LLM_MODEL_NAME: self.llm,
                SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(options),
                SpanAttributes.INPUT_VALUE: prompt,
            })
            if template:
                span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE, template[0])
                span.set_attribute(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, json.dumps(template[1]))
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.llm, "prompt": prompt, "stream": False, "options": options},
                    timeout=300,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                raise RuntimeError(
                    f"Could not reach Ollama at {self.ollama_url}. Start Ollama and pull '{self.llm}'."
                ) from error
            body = response.json()
            reply = body["response"].strip()
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, reply)
            prompt_tokens, completion_tokens = body.get("prompt_eval_count"), body.get("eval_count")
            if prompt_tokens is not None and completion_tokens is not None:
                span.set_attributes({
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL: prompt_tokens + completion_tokens,
                })
            return reply
