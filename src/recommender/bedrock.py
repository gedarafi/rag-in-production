"""Bedrock calls of the recommender: chat in, text or schema-valid JSON out, each one an LLM span.

MODEL_API picks the transport:
  converse  Converse on bedrock-runtime (LocalStack runs an `ollama.<model>` ID on CPU).
  openai    OpenAI-compatible Chat Completions on bedrock-mantle, with response_format json_schema.
  ollama    Ollama's own chat API with `format` (schema-constrained decoding), for the local stack: LocalStack's
            Bedrock emulation drops that, and a small local model needs it to follow the schemas.
  stub      A deterministic stand-in (stub.py) for fast LocalStack and CI runs; no model at all.

Neither LocalStack nor every Bedrock model enforces a JSON schema, so a structured call also puts the
schema in the system prompt, extracts the JSON from the reply, validates it, and asks once more with
the validation error before giving up.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

import jsonschema
from openinference.instrumentation import get_input_attributes, get_llm_attributes, get_output_attributes

from ..tracing import event, fail, tracer

# Bedrock errors that mean "try again later" rather than "this request is broken".
BUSY_ERRORS = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}
BUSY_STATUSES = {429, 503}

SCHEMA_INSTRUCTION = ("\n\nReturn only one JSON object that matches this JSON Schema: no prose, no code fence.\n"
                      "JSON Schema: ")


class ModelError(Exception):
    """A model call that failed in a way the user should hear about; status is the HTTP status to report."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class InvalidOutput(ModelError, ValueError):
    """The model's reply didn't match the schema, even after a correction.

    A ValueError too, so the pipeline treats it like any malformed plan: it falls back where it can.
    """

    def __init__(self, message: str):
        super().__init__(502, message)


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class Converse:
    """Converse on bedrock-runtime, through boto3."""

    system = "aws"

    def __init__(self, client=None, endpoint_url: str | None = None, timeout: float = 60):
        self._client, self.endpoint_url, self.timeout = client, endpoint_url, timeout

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Retries stay with SQS: a failed answer is retried as a whole.
            config = Config(connect_timeout=5, read_timeout=self.timeout, retries={"max_attempts": 2, "mode": "standard"})
            self._client = boto3.client("bedrock-runtime", endpoint_url=self.endpoint_url, config=config)
        return self._client

    def complete(self, model, system, messages, max_tokens, temperature, schema=None, name=None) -> Reply:
        res = self.client.converse(
            modelId=model,
            system=[{"text": system}],
            messages=[{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        text = "".join(block.get("text", "") for block in res["output"]["message"]["content"])
        usage = res.get("usage") or {}
        return Reply(text, usage.get("inputTokens"), usage.get("outputTokens"), usage.get("totalTokens"))


class Mantle:
    """Chat Completions on bedrock-mantle, signed with the caller's AWS credentials.

    boto3 has no bedrock-mantle client, so botocore signs the request and urllib sends it.
    """

    system = "openai"

    def __init__(self, region: str, timeout: float = 60):
        import botocore.session

        self.region, self.timeout = region, timeout
        self.url = f"https://bedrock-mantle.{region}.api.aws/v1/chat/completions"
        self.credentials = botocore.session.get_session().get_credentials()
        # Schemas the endpoint refused in strict mode; they are sent as non-strict json_schema.
        self.loose: set[str] = set()

    def post(self, body: dict) -> dict:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        data = json.dumps(body).encode()
        signed = AWSRequest(method="POST", url=self.url, data=data, headers={"content-type": "application/json"})
        SigV4Auth(self.credentials.get_frozen_credentials(), "bedrock-mantle", self.region).add_auth(signed)
        request = urllib.request.Request(self.url, data=data, headers=dict(signed.headers.items()), method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as res:
            return json.load(res)

    def complete(self, model, system, messages, max_tokens, temperature, schema=None, name=None) -> Reply:
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": max_tokens, "temperature": temperature}
        if schema is not None:
            name = schema_name(schema)
            strict = name not in self.loose
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": name, "strict": strict, "schema": schema}}
            try:
                res = self.post(body)
            except urllib.error.HTTPError as err:
                if err.code != 400 or not strict:
                    raise
                # Strict mode accepts only a subset of JSON Schema; validation below still applies.
                self.loose.add(name)
                body["response_format"]["json_schema"]["strict"] = False
                res = self.post(body)
        else:
            res = self.post(body)
        # A reasoning model that runs out of tokens while thinking returns content null.
        choices = res.get("choices") or [{}]
        usage = res.get("usage") or {}
        return Reply(choices[0].get("message", {}).get("content") or "",
                     usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))


class Ollama:
    """Ollama's /api/chat, with the JSON schema as `format` so the model can only produce valid output."""

    system = "ollama"

    def __init__(self, url: str, timeout: float = 60, context_window: int = 8192):
        self.url, self.timeout, self.context_window = url.rstrip("/"), timeout, context_window

    def complete(self, model, system, messages, max_tokens, temperature, schema=None, name=None) -> Reply:
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages], "stream": False,
                "think": False, "keep_alive": "30m",
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.context_window}}
        if schema is not None:
            body["format"] = schema
        request = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as res:
            data = json.load(res)
        prompt, completion = data.get("prompt_eval_count"), data.get("eval_count")
        total = prompt + completion if prompt is not None and completion is not None else None
        return Reply((data.get("message") or {}).get("content") or "", prompt, completion, total)


def schema_name(schema: dict) -> str:
    """A short stable name for a schema, as OpenAI-style APIs require one."""
    import hashlib

    return "schema_" + hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]


def extract_json(text: str):
    """The first JSON value in a reply, tolerating code fences and prose around it."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            return decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
    raise ValueError("The reply contains no JSON.")


def map_error(err: Exception) -> ModelError:
    """The user-facing error for a failed Bedrock call."""
    if isinstance(err, ModelError):
        return err
    if isinstance(err, urllib.error.HTTPError):
        # bedrock-mantle answers errors with an HTTP status and a JSON body.
        print(f"Bedrock call failed: HTTP {err.code}: {err.read(500).decode(errors='replace')}")
        if err.code in BUSY_STATUSES:
            return ModelError(429, "The assistant is busy. Try again in a moment.")
        return ModelError(502, "The assistant couldn't answer. Try again.")
    # botocore's ClientError carries the AWS error code; timeouts and connection errors don't.
    code = (getattr(err, "response", None) or {}).get("Error", {}).get("Code", "")
    print(f"Bedrock call failed: {code or type(err).__name__}: {err}")
    if code in BUSY_ERRORS:
        return ModelError(429, "The assistant is busy. Try again in a moment.")
    if code:
        return ModelError(502, "The assistant couldn't answer. Try again.")
    return ModelError(504, "The assistant took too long to answer. Try again.")


class BedrockChat:
    """Text and JSON chat with one Bedrock model, traced as LLM spans named after their purpose."""

    def __init__(self, model: str, transport):
        self.model, self.transport = model, transport

    @classmethod
    def from_env(cls) -> "BedrockChat":
        api = os.environ.get("MODEL_API", "converse")
        # Per call; a CPU model on LocalStack needs minutes for a long planning prompt.
        timeout = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "60"))
        if api == "stub":
            from .stub import StubTransport
            transport = StubTransport()
        elif api == "ollama":
            transport = Ollama(os.environ.get("OLLAMA_URL", "http://localhost:11435"), timeout=timeout)
        elif api == "openai":
            transport = Mantle(os.environ["AWS_REGION"], timeout=timeout)
        else:
            # LocalStack's Bedrock, when the CLI runs outside AWS against it.
            transport = Converse(endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL") or None, timeout=timeout)
        return cls(os.environ["MODEL_ID"], transport)

    def text(self, name: str, system: str, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        return self._call(name, system, messages, temperature, max_tokens, None)

    def json(self, name: str, system: str, messages: list[dict], schema: dict, *, temperature: float = 0,
             max_tokens: int = 1024, validate: bool = True) -> str:
        """Schema-valid JSON text; one correction round, then InvalidOutput.

        validate=False returns the reply as it is, for callers with their own validation and retries.
        """
        system = system + SCHEMA_INSTRUCTION + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        text = self._call(name, system, messages, temperature, max_tokens, schema)
        if not validate:
            return text
        try:
            return self.validated(text, schema)
        except (ValueError, jsonschema.ValidationError) as error:
            problem = getattr(error, "message", str(error))
            event("schema_validation_failed", llm=name, error=problem, reply=text)
        correction = [*messages, {"role": "assistant", "content": text},
                      {"role": "user", "content": f"That reply is invalid: {problem} "
                                                  "Return the corrected JSON object only."}]
        text = self._call(name + ".correction", system, correction, temperature, max_tokens, schema)
        try:
            return self.validated(text, schema)
        except (ValueError, jsonschema.ValidationError) as error:
            event("schema_validation_failed", llm=name + ".correction",
                  error=getattr(error, "message", str(error)), reply=text)
            raise InvalidOutput("The assistant returned an invalid response. Try again.") from error

    @staticmethod
    def validated(text: str, schema: dict) -> str:
        value = extract_json(text)
        jsonschema.validate(value, schema)
        return json.dumps(value, ensure_ascii=False)

    def _call(self, name, system, messages, temperature, max_tokens, schema) -> str:
        with tracer.start_as_current_span(name, openinference_span_kind="llm") as span:
            span.set_attributes({
                **get_llm_attributes(
                    provider="aws",
                    system=self.transport.system,
                    model_name=self.model,
                    invocation_parameters={"max_tokens": max_tokens, "temperature": temperature,
                                           **({"json_schema": schema_name(schema)} if schema else {})},
                    input_messages=[{"role": "system", "content": system}, *messages],
                ),
                **get_input_attributes(messages[-1]["content"] if messages else ""),
            })
            try:
                reply = self.transport.complete(self.model, system, messages, max_tokens, temperature, schema, name=name)
            except Exception as err:
                fail(span, err)
                raise map_error(err) from err
            counts = {"prompt": reply.prompt_tokens, "completion": reply.completion_tokens,
                      "total": reply.total_tokens}
            span.set_attributes({
                **get_llm_attributes(
                    output_messages=[{"role": "assistant", "content": reply.text}],
                    token_count={key: count for key, count in counts.items() if count is not None} or None,
                ),
                **get_output_attributes(reply.text),
            })
            if not reply.text.strip():
                fail(span, "empty reply")
                raise ModelError(502, "The assistant returned an empty answer. Try again.")
            return reply.text
