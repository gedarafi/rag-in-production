"""A gate that reviews each message before the recommender pipeline runs.

Rules decide the obvious cases for free: English is the only supported language, small talk
(greetings, thanks, goodbyes, "what can you do?") gets a fixed reply, and prompt injection, secret
probing and code injection get a fixed refusal. Whatever the rules don't decide goes to one small
LLM classification: anime questions and follow-ups pass to the recommender, anything else gets
the fixed reply for its category. The LLM only picks a category, never the reply text, so a
message cannot steer what the user sees. If that call fails, the message passes; the planner
still handles off-topic requests.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm_schemas import SecurityReview, json_schema, parse
from .prompt_library import PromptLibrary, library


@dataclass(frozen=True)
class Verdict:
    category: str
    reply: str
    blocked: bool = False


# A message is small talk only when every word is one of these and one of them names its kind.
GREETINGS = {'hi', 'hello', 'hey', 'heya', 'hiya', 'howdy', 'yo', 'greetings', 'sup', 'morning', 'afternoon',
             'evening'}
THANKS = {'thanks', 'thank', 'thx', 'ty', 'cheers', 'appreciate', 'great', 'awesome', 'perfect', 'nice', 'cool',
          'ok', 'okay', 'got'}
FAREWELLS = {'bye', 'goodbye', 'cya', 'later', 'night', 'farewell', 'care'}
FILLER = {'there', 'all', 'everyone', 'bot', 'assistant', 'friend', 'how', 'are', 'you', 'doing', 'whats', 'what',
          'up', 'good', 'it', 'so', 'much', 'a', 'lot', 'many', 'very', 'see', 'ya', 'soon', 'take', 'for', 'now',
          'and', 'oh'}
SMALL_TALK_PHRASES = {'how are you', 'how are you doing', 'whats up', 'what up'}
# Common function words of other languages that are not English words. Romaji particles
# (no, wa, ga, ...) are left out so Japanese titles such as "Shingeki no Kyojin" stay English.
FOREIGN = {
    'och', 'är', 'jag', 'vad', 'vilka', 'vilken', 'inte', 'mig', 'någon', 'också', 'något', 'ett', 'hej', 'tack',  # sv
    'el', 'los', 'las', 'que', 'qué', 'por', 'para', 'una', 'es', 'cómo', 'quién', 'puedes', 'del', 'con', 'hola',
    'le', 'les', 'des', 'est', 'une', 'qui', 'quoi', 'pour', 'avec', 'je', 'vous', 'sur', 'bonjour',  # fr
    'der', 'das', 'und', 'ist', 'ich', 'wer', 'nicht', 'mit', 'ein', 'eine', 'welche', 'hallo',  # de
    'não', 'você', 'olá', 'het', 'een', 'niet',  # pt, nl
}
ENGLISH = {'the', 'a', 'an', 'is', 'are', 'was', 'what', 'who', 'which', 'how', 'about', 'and', 'of', 'to', 'in',
           'for', 'with', 'me', 'i', 'you', 'like', 'similar', 'recommend', 'watch', 'show', 'some', 'any',
           'give', 'tell', 'should', 'can', 'do', 'does', 'please'}
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
    'unsupported_language': "I only support English. Please ask again in English, for example " + EXAMPLES,
    'off_topic': "I can only help with anime: questions about a show or a character, or recommendations, "
                 "for example " + EXAMPLES,
    'blocked': "I can't help with that. I only answer questions about anime, their characters and "
               "recommendations from the library.",
}


# The LLM's categories are SecurityReview's; PASS goes to the recommender, BLOCKED gets the refusal.
PASS = 'anime'
BLOCKED = {'prompt_injection', 'secrets', 'code_injection', 'harmful'}
REVIEW_SCHEMA = json_schema(SecurityReview)


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", re.sub(r"['’]", "", text.lower()))


def english(text: str, tokens: list[str]) -> bool:
    """Mostly Latin letters, and not more function words of another language than of English."""
    letters = re.findall(r"[^\W\d_]", text)
    latin = sum(1 for letter in letters if letter.isascii() or 'À' <= letter <= 'ɏ')
    if letters and latin * 2 < len(letters):
        return False
    foreign = sum(token in FOREIGN for token in tokens) + len(re.findall(r'[¿¡]', text))
    if tokens and all(token in FOREIGN for token in tokens):
        return False  # "hej", "hola"
    return foreign < 2 or foreign <= sum(token in ENGLISH for token in tokens)


class SecurityAgent:
    """Reviews a message before the pipeline; a verdict means the pipeline need not run."""

    def __init__(self, models=None, prompts: PromptLibrary | None = None) -> None:
        self.models = models
        self.prompts = prompts or library()
        self.last_review = {}

    def review(self, question: str, history: list[dict] | None = None) -> Verdict | None:
        """Rules first; an LLM classification only for messages the rules let through."""
        self.last_review = {'method': 'rules', 'llm_calls': 0}
        verdict = self.rules(question)
        if verdict or not callable(getattr(self.models, 'review_message', None)):
            return verdict
        self.last_review = {'method': 'llm', 'llm_calls': 1}
        context = [{'role': m['role'], 'content': m['content'][:300]} for m in (history or [])[-4:]]
        prompt = self.prompts.render('security_review', history=json.dumps(context, ensure_ascii=False),
                                     question=json.dumps(question, ensure_ascii=False))
        try:
            decision = parse(SecurityReview, json.loads(self.models.review_message(prompt, REVIEW_SCHEMA)))
        except (RuntimeError, ValueError) as error:
            # Fail open: the rules already ran and the planner still handles off-topic requests.
            self.last_review['error'] = str(error)
            return None
        category = decision.category
        self.last_review['category'] = category
        self.last_review['reason'] = decision.reason[:200]
        if category == PASS:
            return None
        if category in BLOCKED:
            return Verdict(category, REPLIES['blocked'], blocked=True)
        return Verdict(category, REPLIES[category])

    def rules(self, question: str) -> Verdict | None:
        text = (question or '').strip()
        tokens = words(text)
        if not tokens and not re.search(r'\d', text):
            return Verdict('empty', REPLIES['empty'])
        for category, patterns in SECURITY.items():
            if any(pattern.search(text) for pattern in patterns):
                # One refusal for every category: the reply never says which rule matched.
                return Verdict(category, REPLIES['blocked'], blocked=True)
        if not english(text, tokens):
            return Verdict('unsupported_language', REPLIES['unsupported_language'])
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
