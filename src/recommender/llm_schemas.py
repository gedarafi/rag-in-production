"""The fixed shapes of structured LLM replies, as Pydantic models.

Each model is both the JSON schema sent with the request (Ollama `format`, Bedrock `response_format`)
and the check of what came back. Field order matters: the model writes the keys in that order.
Checks of meaning (a mention must quote the request, a role must not change) stay with the step
that needs them. Schemas whose choices come from the catalog per request
(tool plans, routing, entity selection) stay hand-built next to their step.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, WithJsonSchema

Intent = Literal['recommendation', 'factual', 'other']
Clauses = Annotated[list[str], Field(max_length=8)]


class Reply(BaseModel):
    """Exactly the declared keys, with JSON types as declared (no "true" for true)."""
    model_config = ConfigDict(extra='forbid', strict=True)


class Decomposition(Reply):
    intent: Intent
    use_previous_context: bool
    entity_queries: Clauses
    required: Clauses
    excluded: Clauses
    preferred: Clauses


class Mention(Reply):
    # The schema must stay exactly as the step was tuned with: Ollama's constrained decoding changes with
    # it, `required` order included. Field order is generation order: a schema-constrained model writes the keys as listed, and a small
    # model copies the request's words reliably only when `text` comes first, before any interpretation.
    text: str
    query: str
    kind: Literal['anime', 'character', 'genre', 'tag', 'description', 'other']
    role: Literal['subject', 'reference', 'required', 'excluded', 'preferred', 'ambiguous']
    # The length bound guides generation; a longer meaning is still accepted.
    meaning: Annotated[str, WithJsonSchema({'type': 'string', 'maxLength': 180})]

    model_config = ConfigDict(json_schema_extra={'required': ['meaning', 'text', 'kind', 'role', 'query']})


class MentionInterpretation(Reply):
    intent: Intent
    use_previous_context: bool
    mentions: Annotated[list[Mention], Field(max_length=16)]


class ConceptRole(Reply):
    role: Literal['required', 'excluded', 'preferred', 'irrelevant']


# Bounds of a numeric request; null for a bound that wasn't requested. (A docstring would enter the schema.)
class NumericRange(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    min: Annotated[float | None, WithJsonSchema({'type': ['number', 'null']})]
    max: Annotated[float | None, WithJsonSchema({'type': ['number', 'null']})]


class FactExtraction(Reply):
    values: list[str]
    unclear: bool
    note: str


class TitleExtraction(Reply):
    titles: list[str]
    unclear: bool
    note: str


class SecurityReview(Reply):
    # anime passes to the recommender; every other category gets the security agent's fixed reply.
    category: Literal['anime', 'greeting', 'thanks', 'farewell', 'help', 'off_topic', 'unsupported_language',
                      'code_injection', 'harmful', 'prompt_injection', 'secrets']
    reason: Annotated[str, WithJsonSchema({'type': 'string', 'maxLength': 200})]


def json_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema, self-contained (nested models inlined), without titles, and with bare
    enums (no "type" beside them), as the hand-written schemas the steps were tuned with."""
    schema = model.model_json_schema()
    definitions = schema.pop('$defs', {})

    def clean(node):
        if isinstance(node, dict):
            if '$ref' in node:
                return clean(definitions[node['$ref'].rsplit('/', 1)[-1]])
            return {key: clean(value) for key, value in node.items()
                    if key != 'title' and not (key == 'type' and 'enum' in node)}
        if isinstance(node, list):
            return [clean(item) for item in node]
        return node

    return clean(schema)


def parse[T: BaseModel](model: type[T], data) -> T:
    """data (parsed JSON) as the model, or a ValueError short enough to send back to the model."""
    try:
        return model.model_validate(data)
    except ValidationError as error:
        problems = ('; '.join(f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in error.errors()))
        raise ValueError(f'Invalid {model.__name__}: {problems}') from None
