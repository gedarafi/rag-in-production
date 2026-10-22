"""A rule-based gate that reviews each message before the recommender pipeline runs.

Small talk (greetings, thanks, goodbyes, "what can you do?") gets a fixed reply, and prompt
injection, secret probing and code injection get a fixed refusal, all without an LLM call,
an embedding or a catalog lookup. Everything else, including a greeting followed by a real
question, passes through unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    category: str
    reply: str
    blocked: bool = False


# A message is small talk only when every word is one of these and one of them names its kind.
GREETINGS = {'hi', 'hello', 'hey', 'heya', 'hiya', 'howdy', 'yo', 'greetings', 'sup', 'hej', 'hejsan', 'hallå',
             'tjena', 'hola', 'buenas', 'morning', 'afternoon', 'evening'}
THANKS = {'thanks', 'thank', 'thx', 'ty', 'cheers', 'tack', 'gracias', 'appreciate', 'great', 'awesome', 'perfect',
          'nice', 'cool', 'ok', 'okay', 'got'}
FAREWELLS = {'bye', 'goodbye', 'cya', 'later', 'night', 'farewell', 'hejdå', 'adios', 'adiós', 'care'}
FILLER = {'there', 'all', 'everyone', 'bot', 'assistant', 'friend', 'how', 'are', 'you', 'doing', 'whats', 'what',
          'up', 'good', 'it', 'so', 'much', 'a', 'lot', 'many', 'very', 'see', 'ya', 'soon', 'take', 'for', 'now',
          'and', 'oh'}
SMALL_TALK_PHRASES = {'how are you', 'how are you doing', 'whats up', 'what up'}
HELP = re.compile(r"^(?:help|help me|what can you do|what do you do|who are you|what are you|how does this work"
                  r"|how do i use (?:this|you)|what can i ask(?: you)?|what do you know)$")

SECURITY = {
    'prompt_injection': [
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,30}"
        r"\b(?:previous|prior|above|earlier|preceding|system|safety|original|your|all)\b.{0,15}"
        r"\b(?:instructions?|prompts?|rules|guidelines|directives)\b",
        r"\b(?:system|developer|hidden|initial|original)\s+(?:prompt|instructions?)\b|\bsystem message\b",
        r"\b(?:reveal|show|print|repeat|output|leak|tell me|what are)\b.{0,20}"
        r"\byour\s+(?:\w+\s+)?(?:prompt|instructions|rules|configuration)\b",
        r"\b(?:jailbreak|dan mode|developer mode|do anything now)\b",
        r"\byou are no longer\b",
        r"\b(?:pretend|act|behave|roleplay)\b.{0,30}\b(?:no|without)\s+(?:rules|restrictions|filters|limits|guidelines)\b",
        r"<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?inst\]|(?m:^\s*(?:system|assistant)\s*:)",
    ],
    # Asking for a secret, not mentioning one: "an anime about a hacker who steals passwords" passes.
    'secrets': [
        r"\b(?:give|show|tell|reveal|print|send|share|dump|leak|list|display|output|read|whats|what is|what are)\b"
        r".{0,40}\b(?:(?:api|access|secret|private)[\s_-]?keys?|credentials|passwords?|jwt"
        r"|(?:auth(?:entication)?|bearer|session|access)[\s_-]?tokens?|environment variables|env vars)\b",
        r"(?:^|\s)\.env\b|/etc/(?:passwd|shadow)\b",
    ],
    'code_injection': [
        r"\b(?:drop|truncate|alter)\s+table\b|\bdelete\s+from\s+\w+\s*(?:where\b|;)|\bunion\s+(?:all\s+)?select\b",
        r";\s*--|\bor\s+1\s*=\s*1\b|'\s*or\s*'1'\s*=\s*'1",
        r"\brm\s+-rf\b|\$\([^)]*\)|<\s*script\b|javascript:",
    ],
}
SECURITY = {category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in SECURITY.items()}

EXAMPLES = ('"What should I watch after Steins;Gate?", "Who is Lelouch?" or '
            '"Recommend three short comedy anime".')
REPLIES = {
    'empty': "Ask me about an anime or a character, or for recommendations, for example " + EXAMPLES,
    'greeting': "Hi! I'm the anime assistant. Ask me about an anime or a character, or for recommendations, "
                "for example " + EXAMPLES,
    'thanks': "Glad to help! Ask me anything else about anime whenever you like.",
    'farewell': "Bye! Come back any time for more anime recommendations.",
    'help': "I answer questions about anime and their characters and recommend titles from an AniList snapshot. "
            "I can describe a show or a character, list its main cast, give facts such as episodes, year, studio "
            "or score, and recommend similar titles or ones matching genres, tags, years or length. "
            "Try " + EXAMPLES,
    'blocked': "I can't help with that. I only answer questions about anime, their characters and "
               "recommendations from the library.",
}


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", re.sub(r"['’]", "", text.lower()))


class SecurityAgent:
    """Reviews a message before the pipeline; a verdict means the pipeline need not run."""

    def review(self, question: str) -> Verdict | None:
        text = (question or '').strip()
        tokens = words(text)
        if not tokens and not re.search(r'\d', text):
            return Verdict('empty', REPLIES['empty'])
        for category, patterns in SECURITY.items():
            if any(pattern.search(text) for pattern in patterns):
                # One refusal for every category: the reply never says which rule matched.
                return Verdict(category, REPLIES['blocked'], blocked=True)
        normalized = ' '.join(tokens)
        if HELP.match(normalized):
            return Verdict('help', REPLIES['help'])
        found = set(tokens)
        if re.search(r'\d', text) or len(tokens) > 8 or not found <= GREETINGS | THANKS | FAREWELLS | FILLER:
            return None
        if found & FAREWELLS:
            return Verdict('farewell', REPLIES['farewell'])
        if found & THANKS:
            return Verdict('thanks', REPLIES['thanks'])
        if found & GREETINGS or normalized in SMALL_TALK_PHRASES:
            return Verdict('greeting', REPLIES['greeting'])
        return None
