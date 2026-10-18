"""Intentionally rebuild 72 authored requests and frozen local catalog references.

This refreshes evaluation labels; evaluation itself never rebuilds them.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommender.recommendation_evaluation import Catalog, digest, load_dataset
from src.recommender.factual_evaluation import write_csv

DEST = ROOT / 'data/evaluation/recommendations'
DATABASE = ROOT / 'data/documents/anilist/catalog.sqlite3'


def c(field, op, value):
    return {'field': field, 'op': op, 'value': value}


# Each family has one explicit and two indirect naturalistic requests. Mood labels
# are documented operational proxies, not a claim that everyone feels the same.
# Each complete question is authored individually, not expanded from a template.
FAMILIES = [
    ('comfort', 'tags', [c('tags', 'any', ['Iyashikei'])],
     'Comfort and low-stress daily life are operationalized using the Iyashikei tag; this cannot guarantee a viewer mood.', [
         'Can you recommend three iyashikei anime for winding down?',
         'Work has been exhausting. What are four gentle anime I can put on to relax and feel looked after?',
         'I need five quiet, comforting watches where small everyday moments are the main attraction. What would you pick?']),
    ('tearjerker', 'tags', [c('tags', 'any', ['Tragedy'])],
     'An emotionally devastating or unhappy story is approximated by Tragedy, not by a nonexistent Sad tag.', [
         'Give me three anime with tragedy as a major theme.',
         'I want a good cry tonight. Which four anime tell heartbreaking stories where things really do go wrong?',
         'What are five anime that will leave me emotionally wrecked rather than wrapping everything up happily?']),
    ('laughter', 'tags', [c('tags', 'any', ['Surreal Comedy', 'Slapstick'])],
     'Absurd and physical comedy are alternative acceptable routes to the requested laughter.', [
         'Recommend three anime with surreal comedy or slapstick.',
         'I could use a laugh. What are four shows full of ridiculous situations and over-the-top physical gags?',
         'Give me five anime where the joke is often something completely absurd happening on screen.']),
    ('time', 'tags', [c('tags', 'any', ['Time Manipulation'])],
     'Rewinding, looping, and changing time map to Time Manipulation.', [
         'Which three anime would you recommend for the Time Manipulation tag?',
         'What are four anime where someone can rewind events or change what happens by going back in time?',
         'I love stories where the same stretch of time can play out differently. Can you suggest five anime like that?']),
    ('found_family', 'tags', [c('tags', 'any', ['Found Family'])],
     'Unrelated people forming a family-like bond map to Found Family.', [
         'Recommend three anime with a found family theme.',
         'What are four anime about people who are not related but slowly become each other\'s home?',
         'I want five shows where a group of outsiders end up treating one another like family. Any ideas?']),
    ('cyberpunk', 'tags', [c('tags', 'any', ['Cyberpunk'])],
     'High technology and a gritty unequal future are a Cyberpunk proxy.', [
         'Can I get three cyberpunk anime recommendations?',
         'Which four anime have grimy high-tech cities, body upgrades, and a future that feels pretty broken?',
         'Suggest five anime set in a harsh wired-up future where advanced technology has not made ordinary life better.']),
    ('detectives', 'tags', [c('tags', 'any', ['Detective'])],
     'A central investigator solving cases maps to Detective.', [
         'Give me three anime that carry the Detective tag.',
         'What are four good anime about investigators piecing together evidence to solve cases?',
         'I want five anime where following clues and figuring out who did it drives the story. What should I try?']),
    ('space_military', 'multiple_tags', [c('tags', 'any', ['Space']), c('tags', 'any', ['Military'])],
     'Both space setting and military organization must hold for each recommendation.', [
         'Recommend three anime with both Space and Military themes.',
         'Which four anime follow armed forces fighting beyond Earth?',
         'I would like five anime about organized fleets or soldiers carrying out missions in outer space.']),
    ('comfort_exclusion', 'tag_exclusions', [c('tags', 'any', ['Iyashikei']), c('tags', 'none', ['Tragedy'])],
     'Iyashikei is required; a Tragedy rank of 50 or higher violates the exclusion. Missing tags only mean no catalog evidence.', [
         'Give me three iyashikei recommendations without a strong Tragedy tag.',
         'I need four gentle, soothing anime to decompress with. Please skip stories built around devastating loss or unhappy endings.',
         'What are five calm everyday-life anime for a rough week? I do not want the story to turn into a tragedy.']),
    ('fantasy', 'genres', [c('genres', 'any', ['Fantasy'])],
     'Magic and impossible worlds map to the Fantasy genre.', [
         'Recommend three anime in the fantasy genre.',
         'What are four anime that will take me into a world of magic and impossible creatures?',
         'Give me five anime where the world works by enchanted rules instead of ordinary reality.']),
    ('romance', 'genres', [c('genres', 'any', ['Romance'])],
     'A central falling-in-love story maps to Romance.', [
         'What are three romance anime you would recommend?',
         'I want four anime where falling in love is the main reason to watch. What should I start with?',
         'Can you suggest five anime centered on two people developing feelings for each other?']),
    ('horror', 'genres', [c('genres', 'any', ['Horror'])],
     'Frightening viewing maps to Horror, not every dark or sad story.', [
         'Give me three anime from the horror genre.',
         'Which four anime would make a good lights-off viewing session if I want to be genuinely scared?',
         'I am in the mood for five creepy anime that try to frighten the viewer. What do you suggest?']),
    ('comedy', 'genres', [c('genres', 'any', ['Comedy'])],
     'The broad request to laugh maps to Comedy rather than a specific joke style.', [
         'Recommend three comedy anime.',
         'What are four anime where making the audience laugh is the main point?',
         'I need five funny shows to brighten up the evening. Which anime would you pick?']),
    ('recent', 'release_year', [c('year', 'gte', 2020)],
     'First release year of the specific catalog entry, inclusive from 2020.', [
         'Recommend three anime first released in 2020 or later.',
         'Which four anime that debuted from the start of this decade onward should I watch? By this decade I mean the 2020s.',
         'Give me five anime that first came out no earlier than 2020. I want something from the newer wave.']),
    ('nineties', 'release_year', [c('year', 'gte', 1990), c('year', 'lte', 1999)],
     'The original entry must first release in the inclusive 1990-1999 interval.', [
         'What are three anime released between 1990 and 1999, inclusive?',
         'I am feeling nostalgic for the nineties. Can you suggest four anime that actually debuted back then?',
         'Give me five anime that first came out after the eighties ended but before the new millennium began.']),
    ('dates', 'release_date', [c('start_date', 'gte', '2018-01-01'), c('start_date', 'lte', '2019-12-31')],
     'Uses complete known start dates, inclusive; partial dates are unknown and fail.', [
         'Recommend three anime with a first release date from 2018-01-01 through 2019-12-31.',
         'What are four anime that debuted during 2018 or 2019? I am catching up on what I missed just before 2020.',
         'I missed two whole years of releases before 2020. Give me five anime that first arrived in 2018 or 2019.']),
    ('kyoani', 'studios', [c('studios', 'any', ['Kyoto Animation'])],
     'Only studios marked main=true AND animation=true in the local export count.', [
         'Which three anime made by Kyoto Animation would you recommend?',
         'I loved the animation in Violet Evergarden. Suggest four anime from the same animation studio.',
         'Can you give me five anime by the team at KyoAni?']),
    ('trigger', 'studios', [c('studios', 'any', ['Trigger'])],
     'Trigger must be a main animation studio; perceived visual similarity is insufficient.', [
         'Recommend three anime animated by Studio Trigger.',
         'I like the studio behind Kill la Kill. What are four anime from that animation studio that I could watch?',
         'Give me five anime made by the animation company that produced Promare.']),
    ('movies', 'formats', [c('format', 'any', ['MOVIE'])],
     'Movie format only; no runtime cap is implied.', [
         'Recommend three anime movies.',
         'I do not want to start a series. What are four animated films from the anime catalog I could watch?',
         'Give me five anime made as films instead of episodic shows.']),
    ('tv', 'formats', [c('format', 'any', ['TV'])],
     'TV format only; web-only series and TV_SHORT are different catalog formats.', [
         'Recommend three anime released as TV series, not TV shorts.',
         'What are four regular television anime series I could follow episode by episode?',
         'I want five anime that originally aired as standard TV shows. What would you choose?']),
    ('manga', 'source_material', [c('source_material', 'any', ['MANGA'])],
     'MANGA source medium is required; popularity or manga-like style is insufficient.', [
         'Recommend three anime adapted from manga.',
         'What are four anime that began as Japanese comics before getting animated?',
         'Give me five anime where I could go read the original manga afterward.']),
    ('short_series', 'length', [c('format', 'any', ['TV']), c('episodes', 'lte', 12)],
     'Episode count belongs to a single catalog entry, not the complete franchise.', [
         'Recommend three TV anime with at most 12 episodes in the specific series you suggest.',
         'I only have room for a dozen episodes per show. Which four TV anime entries fit that limit?',
         'Give me five regular TV anime I could finish within twelve episodes each. Name the specific season if needed.']),
    ('short_episodes', 'length', [c('format', 'any', ['TV', 'TV_SHORT']), c('duration', 'lte', 15)],
     'Duration is minutes per episode, not total series runtime.', [
         'Recommend three TV anime or TV shorts whose episodes run no more than 15 minutes each.',
         'My break is only a quarter of an hour. What are four television anime with episodes that fit into it?',
         'I want five anime made for TV that I can watch one episode at a time in fifteen minutes or less.']),
    ('mixed', 'mixed_constraints', [c('genres', 'any', ['Fantasy']), c('year', 'gte', 2020), c('source_material', 'any', ['MANGA'])],
     'Fantasy genre, release from 2020, and manga source are jointly required.', [
         'Recommend three fantasy anime released from 2020 onward that were adapted from manga.',
         'What are four newer anime about magical worlds that started as comics? Please keep the debut year to 2020 or later.',
         'Give me five anime with enchanted worlds, an original manga to read, and a first release no earlier than 2020.']),
]


def complete_date(search):
    values = [search.get('startDate_' + part) for part in ('year', 'month', 'day')]
    try:
        return date(*(int(v) for v in values)).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def build():
    if not DATABASE.is_file():
        raise SystemExit('AniList catalog is not prepared. Run `uv run recommender build` first, then rerun this command. Offline notebook inspection does not need this export.')
    DEST.mkdir(parents=True, exist_ok=True)
    policy = json.loads((ROOT / 'scripts/recommendation_semantic_policy.json').read_text(encoding='utf-8'))
    wanted_tags = {t for _, _, cs, _, _ in FAMILIES for item in cs if item['field'] == 'tags' for t in item['value']}
    wanted_tags.update(t for rule in policy['rules'] for branch in rule['alternatives']
                       for item in branch['constraints'] if item['field'] == 'tags' for t in item['value'])
    found_tags = set()
    with closing(sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        meta = dict(db.execute('SELECT * FROM meta'))
        with (DEST / 'catalog_reference.jsonl').open('w', encoding='utf-8', newline='\n') as out:
            for identifier, data, metadata in db.execute(
                    'SELECT a.id,a.data,m.data FROM anime a LEFT JOIN anime_metadata m ON m.anime_id=a.id ORDER BY a.id'):
                r, m = json.loads(data), json.loads(metadata or '{}')
                search = m.get('search_data', {})
                ranks = {}
                for t in search.get('tag_details', []):
                    if t.get('name'):
                        rank = t.get('rank')
                        if type(rank) not in (int, float) or not 0 <= rank <= 100:
                            raise ValueError(f'Invalid tag rank: {identifier}/{t}')
                        ranks[t['name']] = rank
                        found_tags.add(t['name'])
                record = {'id': identifier, 'title': r['title'], 'aliases': r.get('aliases', []),
                          'tag_ranks': ranks, 'genres': r.get('genres', []), 'year': r.get('year'),
                          'start_date': complete_date(search), 'format': r.get('format'),
                          'source_material': m.get('source_material') or None,
                          'episodes': r.get('episodes'), 'duration': r.get('duration'),
                          'studios': [s['name'] for s in m.get('studios', []) if s.get('main') and s.get('animation')]}
                out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    if not wanted_tags <= found_tags:
        raise ValueError('Unknown/missing tag rank definitions: ' + repr(wanted_tags - found_tags))
    catalog = Catalog(DEST / 'catalog_reference.jsonl')
    questions = []
    for family, category, constraints, rationale, prompts in FAMILIES:
        for index, prompt in enumerate(prompts):
            expected = deepcopy(constraints)
            for rule in policy['rules']:
                if family in rule['families'] and index + 1 in rule['prompt_numbers']:
                    expected[rule['constraint_index']]['alternatives'] = deepcopy(rule['alternatives'])
            matches = [r for r in catalog.records.values() if catalog.fits({'constraints': expected}, r)[1]]
            matches.sort(key=lambda r: (-sum(catalog.fits({'constraints': expected}, r)[0]), r['id']))
            k = index + 3
            if len(matches) < k:
                raise ValueError(f'Not enough eligible entries: {family} has {len(matches)} matches')
            questions.append({'question_id': f'{family}_{index + 1:02d}', 'category': category,
                              'intent_family': family, 'wording': 'explicit' if index == 0 else 'indirect',
                              'question_origin': 'authored_naturalistic', 'question': prompt, 'k': k,
                              'constraints': expected, 'label_rationale': rationale,
                              'semantic_policy': 'reviewed_alternatives' if any(c.get('alternatives') for c in expected) else 'literal',
                              'strict_catalog_matches': len(matches),
                              'reference_examples': [{'id': r['id'], 'title': r['title']} for r in matches[:3]]})
    (DEST / 'questions.jsonl').write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in questions), encoding='utf-8')
    write_csv(DEST / 'questions.csv', questions, list(questions[0]))
    (DEST / 'semantic_policy.json').write_text(json.dumps(policy, indent=2) + '\n', encoding='utf-8')
    manifest = {'dataset_version': 2, 'question_count': len(questions), 'catalog_count': len(catalog.records),
                'catalog_tag_count': len(found_tags),
                'requests_with_alternatives': sum(q['semantic_policy'] == 'reviewed_alternatives' for q in questions),
                'origin': 'Individually authored naturalistic synthetic English requests; not real user logs.',
                'source': 'Imported local AniList catalog and anime_metadata.search_data; no live website lookups.',
                'catalog_meta': {k: meta[k] for k in ('snapshot', 'production_metadata')},
                'tag_threshold': 50, 'by_category': dict(Counter(q['category'] for q in questions)),
                'by_wording': dict(Counter(q['wording'] for q in questions)),
                'files_sha256': {name: digest(DEST / name) for name in ('questions.jsonl', 'questions.csv', 'catalog_reference.jsonl', 'semantic_policy.json')}}
    (DEST / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    load_dataset(DEST)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__':
    build()
