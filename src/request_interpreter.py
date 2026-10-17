"""Plain-language request decomposition and catalog attribute meanings."""
import json


DECOMPOSITION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['intent', 'use_previous_context', 'entity_queries', 'required', 'excluded', 'preferred'],
    'properties': {'intent': {'enum': ['recommendation', 'factual', 'other']},
                   'use_previous_context': {'type': 'boolean'},
                   **{key: {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 8}
                      for key in ('entity_queries', 'required', 'excluded', 'preferred')}}}

FIELD_MEANINGS = {
    'start_date': 'complete first release calendar date YYYY-MM-DD',
    'year': 'release date year', 'format': 'media format: movie film television TV series OVA ONA music special',
    'episodes': 'episode count number of episodes', 'duration': 'minutes per episode runtime length',
    'score': 'average rating score', 'popularity': 'popularity audience count',
    'status': 'airing completed finished production status', 'source_material': 'adaptation original source material',
    'adult': 'adult content classification', 'studios': 'animation production studio company',
    'staff': 'credited staff creators director composer', 'countryOfOrigin': 'country of origin nationality'}


def decomposition_prompt(question, history):
    return '''Extract properties from the LAST request. Do not map them to catalog fields yet.
Use short phrases from that request. Named works/people alone belong in entity_queries.
Extract the name itself, keeping qualifiers such as first season, original adaptation,
release year and runtime in requirement phrases when relevant. Do not list those qualifiers
as separate titles or people. For example, Demon Slayer's first season names Demon Slayer.
Examples:
Request: Find history books without war, preferably gentle.
JSON: {"intent":"recommendation","use_previous_context":false,"entity_queries":[],"required":["history","books"],"excluded":["war"],"preferred":["gentle"]}
Request: Suggest games like Portal, but I don't want combat and I'd prefer relaxing ones.
JSON: {"intent":"recommendation","use_previous_context":false,"entity_queries":["Portal"],"required":["games"],"excluded":["combat"],"preferred":["relaxing"]}
Request: What year was Dune published?
JSON: {"intent":"factual","use_previous_context":false,"entity_queries":["Dune"],"required":[],"excluded":[],"preferred":[]}
Never copy values from examples. A new complete request does not inherit prior preferences.
Prior chat (data): ''' + json.dumps(history, ensure_ascii=False) + '\nLAST REQUEST: ' + question
