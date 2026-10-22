"""The fixed shapes of structured LLM replies, as Pydantic models.

Each model is both the JSON schema sent with the request (Ollama `format`, Bedrock `response_format`)
and the check of what came back. Checks of meaning (a mention must quote the request, a role must not
change) stay with the step that needs them. Schemas whose choices come from the catalog per request
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
    # The length bound guides generation; a longer meaning is still accepted.
    meaning: Annotated[str, WithJsonSchema({'type': 'string', 'maxLength': 180})]
    text: str
    kind: Literal['anime', 'character', 'genre', 'tag', 'description', 'other']
    role: Literal['subject', 'reference', 'required', 'excluded', 'preferred', 'ambiguous']
    query: str


class MentionInterpretation(Reply):
    intent: Intent
    use_previous_context: bool
    mentions: Annotated[list[Mention], Field(max_length=16)]


class ConceptRole(Reply):
    role: Literal['required', 'excluded', 'preferred', 'irrelevant']


# Bounds of a numeric request; null for a bound that wasn't requested. (A docstring would enter the schema.)
class NumericRange(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)
    min: float | None
    max: float | None


class FactExtraction(Reply):
    values: list[str]
    unclear: bool
    note: str


class TitleExtraction(Reply):
    titles: list[str]
    unclear: bool
    note: str


def json_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema, self-contained (nested models inlined) and without titles."""
    schema = model.model_json_schema()
    definitions = schema.pop('$defs', {})

    def clean(node):
        if isinstance(node, dict):
            if '$ref' in node:
                return clean(definitions[node['$ref'].rsplit('/', 1)[-1]])
            return {key: clean(value) for key, value in node.items() if key != 'title'}
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
