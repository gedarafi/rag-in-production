"""A Bedrock transport that replies from a script, for testing the recommender's model steps."""
from src.recommender.bedrock import BedrockChat, Reply
from src.recommender.providers import Models


class ScriptedTransport:
    """Returns the given replies in order (the last one repeats) and records every request."""

    system = "aws"

    def __init__(self, *replies, error=None):
        self.replies, self.error, self.calls = list(replies), error, []

    def complete(self, model, system, messages, max_tokens, temperature, schema=None, name=None):
        self.calls.append({"model": model, "system": system, "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "schema": schema})
        if self.error:
            raise self.error
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Reply(text, 10, 5, 15)


def scripted_models(*replies, embedder=None) -> tuple[Models, ScriptedTransport]:
    transport = ScriptedTransport(*replies)
    return Models(chat=BedrockChat("test-model", transport), embedder=embedder), transport
