"""Plain-language request decomposition and catalog attribute meanings."""
import json

from .llm_schemas import Decomposition, json_schema
from .prompt_library import library


DECOMPOSITION_SCHEMA = json_schema(Decomposition)

FIELD_MEANINGS = {
    'start_date': 'complete first release calendar date YYYY-MM-DD',
    'year': 'release date year', 'format': 'media format: movie film television TV series OVA ONA music special',
    'episodes': 'episode count number of episodes', 'duration': 'minutes per episode runtime length',
    'score': 'average rating score', 'popularity': 'popularity audience count',
    'status': 'airing completed finished production status', 'source_material': 'adaptation original source material',
    'adult': 'adult content classification', 'studios': 'animation production studio company',
    'staff': 'credited staff creators director composer', 'countryOfOrigin': 'country of origin nationality'}


def decomposition_prompt(question, history, prompts=None):
    return (prompts or library()).render('decompose_request', history=json.dumps(history, ensure_ascii=False),
                                         question=question)
