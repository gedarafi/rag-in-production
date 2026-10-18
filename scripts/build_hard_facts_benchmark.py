"""Build the authored 180-question benchmark and freeze labels from local AniList.

Run from the repository root: .venv/Scripts/python.exe scripts/build_hard_facts_benchmark.py
Rebuilding intentionally refreshes labels and the snapshot manifest; review the diff.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommender.factual_evaluation import database_values, load_questions

DEST = ROOT / 'data/evaluation/hard_facts'
DATABASE = ROOT / 'data/documents/anilist/catalog.sqlite3'

# Individually authored prompts, not scraped user logs. Each anime/field pair is unique.
# IDs are AniList IDs, not MAL IDs. Wording deliberately specifies seasons and media.
PROMPTS = {
    'year': '''16498|What year did the first season of Attack on Titan start airing?
101922|When did Demon Slayer first come out? Just the year is fine.
113415|Which year was the first JUJUTSU KAISEN TV season released?
1535|When was Death Note first released as an anime? I'd like the year.
21459|What was the premiere year for My Hero Academia season one?
21087|In what year did One-Punch Man make its anime debut?
20605|I'm trying to place Tokyo Ghoul on a timeline. What year did its original TV anime begin?
5114|What year did Fullmetal Alchemist: Brotherhood first air?
20|What year did the original Naruto anime begin?
20954|Which year was the anime film A Silent Voice released?
21519|What is the release year of the movie Your Name?
127230|When was Chainsaw Man's first anime season released? I only need the year.
101759|What year marked the start of The Promised Neverland anime?
20755|Can you give me the year Assassination Classroom's first season premiered?
21507|What year did the Mob Psycho 100 anime originally launch?
20665|Which year did Your Lie in April begin its TV run?
1735|What is Naruto: Shippuden's first airing year?
21234|In which year did the ERASED anime premiere?
9253|What year was the original Steins;Gate TV series released?
20464|When did Haikyu's first TV season start? Please give the year.
21827|What release year is listed for the Violet Evergarden TV series?
101348|Which year did Vinland Saga's first season debut?
30|What was the original broadcast year of Neon Genesis Evangelion?
199|What year did Spirited Away come out?
1|What year did the Cowboy Bebop TV anime first air?
154587|In what year did Frieren: Beyond Journey's End start airing?''',
    'episodes': '''11061|How many episodes are in Hunter x Hunter's 2011 adaptation?
20958|How long is Attack on Titan season two in episodes?
11757|What's the episode count for the first Sword Art Online TV series?
99147|How many episodes does Attack on Titan Season 3 have, excluding Part 2?
21856|How many episodes would I need to watch to finish My Hero Academia season two?
110277|What's the total episode count listed for Attack on Titan Final Season, without Part 2?
21355|How many episodes make up the original Re:ZERO TV season?
104578|Can you tell me the number of episodes in Attack on Titan Season 3 Part 2?
100166|How many episodes were made for My Hero Academia's third season?
97940|How many episodes does the Black Clover TV anime have?
101291|What's the length of Rascal Does Not Dream of Bunny Girl Senpai in episodes?
140960|How many episodes are in the first SPY x FAMILY season, excluding Part 2?
101921|How many episodes does the first Kaguya-sama: Love is War season contain?
105333|What's the episode total for Dr. STONE season one?
124080|How many episodes are there in the original Horimiya TV anime?
4224|What episode count does the catalog give for Toradora!?
19815|How many episodes would finishing No Game, No Life's TV series take?
20613|What's the complete TV episode count for Akame ga Kill!?
20447|How many episodes were in Noragami's first TV season?
20789|How many episodes aired in the first season of The Seven Deadly Sins?
269|How many episodes does the original Bleach series have, before Thousand-Year Blood War?
21202|How many episodes does the first KONOSUBA TV season run for?
1575|What is the number of episodes in Code Geass: Lelouch of the Rebellion's first season?
20623|How many episodes are in Parasyte -the maxim- altogether?
99423|What's the TV episode count of DARLING in the FRANXX?
20931|How many episodes should I expect from Death Parade?''',
    'duration': '''101922|About how many minutes is one episode of Demon Slayer's first season?
113415|What's the listed runtime per episode of JUJUTSU KAISEN season one?
1535|How long does a single Death Note episode run, in minutes?
21459|What is the episode duration for My Hero Academia's first season?
11061|How much time should I set aside for one episode of Hunter x Hunter (2011)? Give minutes.
21087|What's the catalog's per-episode runtime for the original One-Punch Man season?
20605|How many minutes does a typical Tokyo Ghoul season-one episode last?
5114|What running time per episode is listed for Fullmetal Alchemist: Brotherhood?
20|How long is an episode of the original Naruto anime, in minutes?
11757|For Sword Art Online's first TV series, what's the duration of one episode?
127230|How many minutes per episode should I expect for Chainsaw Man's first season?
101759|What runtime does the catalog list for each episode of The Promised Neverland season one?
20755|How long is a standard episode of Assassination Classroom's first season?
21507|Can you give Mob Psycho 100 season one's episode length in minutes?
21355|What is the listed length of an episode of the first Re:ZERO season?
20665|How much viewing time, in minutes, does one Your Lie in April episode take?
1735|What's the standard episode runtime listed for Naruto: Shippuden?
21234|How many minutes is an ERASED episode supposed to run?
9253|What duration per episode is recorded for the original Steins;Gate series?
20464|How long does one first-season Haikyu episode last according to the catalog?
21827|What's Violet Evergarden's TV runtime per episode, in minutes?
101348|For Vinland Saga season one, how many minutes long is each episode listed as?
30|What's the recorded episode length for Neon Genesis Evangelion?
1|How many minutes does a Cowboy Bebop TV episode run?
154587|What per-episode duration is listed for Frieren: Beyond Journey's End?
98314|How long is an episode of Kakegurui's first season?''',
    'format': '''20954|Is A Silent Voice cataloged as a movie or a TV series?
21519|What release format is Your Name listed under?
112151|Is Demon Slayer: Mugen Train's movie entry a film, TV series, or special?
199|How is Spirited Away classified by release format?
131573|Does JUJUTSU KAISEN 0 belong to the movie or TV category?
145064|What kind of anime release is JUJUTSU KAISEN Season 2: TV, OVA, or movie?
142329|Is Demon Slayer's Entertainment District Arc listed as television or a film?
105310|What format does the first Fire Force anime season use?
14719|Is the 2012 JoJo's Bizarre Adventure adaptation a TV series or an OVA?
97668|What release-format category is One-Punch Man Season 2 in?
101280|Is That Time I Got Reincarnated as a Slime's original adaptation a TV anime or movie?
99263|Which release format is recorded for The Rising of the Shield Hero season one?
108465|Is the first Mushoku Tensei: Jobless Reincarnation entry a TV series or a special?
9919|How does the catalog categorize the original Blue Exorcist anime's format?
20594|Is Sword Art Online II a television release or a film?
151807|What is Solo Leveling's first anime season classified as: TV, movie, or OVA?
101338|What release format is Mob Psycho 100 II?
98659|Does Classroom of the Elite's first anime season have a TV or OVA format?
120120|What type of release is the original Tokyo Revengers anime?
6547|Is Angel Beats! listed as a TV series, movie, or ONA?
131681|Which format does Attack on Titan Final Season Part 2 fall under?
112641|Is Kaguya-sama: Love is War's second season a television anime or a special?
20997|What is the release format of Charlotte's main anime entry?
132405|Is the first My Dress-Up Darling anime a TV production or a film?
10620|What format is The Future Diary's main anime series recorded in?
21311|How is Bungo Stray Dogs season one classified: TV series, OVA, or movie?''',
    'source_material': '''16498|What type of source material was Attack on Titan adapted from?
20954|Was A Silent Voice based on a manga, a novel, or an original screenplay?
21519|Is Your Name an original anime work or an adaptation of another medium?
127230|What medium did Chainsaw Man originate in before its anime adaptation?
101759|What is the original source medium for The Promised Neverland anime?
21355|Did Re:ZERO's anime adapt a light novel, manga, or something else?
9253|What kind of work was Steins;Gate adapted from?
101291|What's the source-material category for Rascal Does Not Dream of Bunny Girl Senpai?
140960|What was SPY x FAMILY adapted from originally?
101921|Does the first Kaguya-sama: Love is War anime come from a manga or light novel?
105333|Which source medium underlies Dr. STONE's anime?
21827|What sort of source material is listed for the Violet Evergarden TV anime?
124080|Is Horimiya's anime based on a manga, game, or original story?
4224|What medium is credited as Toradora!'s source material?
19815|Was No Game, No Life adapted from a light novel or a manga?
21202|What is the catalog's source-material label for KONOSUBA's first season?
1575|Is Code Geass: Lelouch of the Rebellion an original anime or an adaptation?
99423|Where does DARLING in the FRANXX fall in terms of source material: original, manga, or novel?
30|What source-material category is recorded for Neon Genesis Evangelion?
199|Was Spirited Away adapted from a previously published work or made as an original anime?
1|What is the source type of the Cowboy Bebop TV series?
20931|Is Death Parade an original anime, a manga adaptation, or a novel adaptation?
101280|What medium was That Time I Got Reincarnated as a Slime's first anime adapted from?
99263|Which kind of source work did The Rising of the Shield Hero adapt?
108465|What source medium is listed for Mushoku Tensei: Jobless Reincarnation?
6547|Was Angel Beats! created as an original anime or adapted from another medium?''',
    'main_animation_studios': '''16498|Which studio handled the main animation for Attack on Titan season one?
101922|Who did the main animation on Demon Slayer's first season?
113415|Which animation studio made the first JUJUTSU KAISEN TV season?
1535|What studio is credited with the main animation of Death Note?
21459|Who was the main animation studio behind My Hero Academia's debut season?
11061|Which studio animated Hunter x Hunter's 2011 TV adaptation?
21087|Who handled One-Punch Man season one's main animation?
20605|Which main animation studio is listed for the first Tokyo Ghoul season?
5114|What studio animated Fullmetal Alchemist: Brotherhood?
20|Which company was responsible for the main animation of the original Naruto series?
20954|Who animated the A Silent Voice film? I mean the animation studio.
21519|Which studio is credited as Your Name's main animation studio?
127230|Who produced the main animation for Chainsaw Man's first season?
21507|What animation studio worked on Mob Psycho 100's first season?
21355|Which studio animated the original Re:ZERO TV season?
20665|What is the main animation studio for Your Lie in April?
21234|Which studio animated ERASED?
9253|Which studio made the animation for the original Steins;Gate series?
20464|Which main animation studio does the catalog credit for Haikyu season one?
140960|Can you list all main animation studios credited for SPY x FAMILY's first season?
21827|Which animation studio was behind the Violet Evergarden TV series?
101348|Who animated Vinland Saga season one, rather than its later seasons?
199|What animation studio made Spirited Away?
1|Which studio is responsible for Cowboy Bebop's TV animation?
97668|Who animated One-Punch Man's second season?''',
    'main_characters': '''16498|Who are all the characters marked as main in Attack on Titan's first-season cast?
101922|Can you list the main characters of Demon Slayer season one?
113415|Who makes up the main cast of JUJUTSU KAISEN's first season? Character names only.
1535|Which Death Note characters have a main role in the catalog?
21459|Who are the main characters listed for My Hero Academia season one?
11061|Give me the complete main-character list for Hunter x Hunter (2011).
21087|Who is in One-Punch Man season one's main character cast?
20605|Which characters are classified as main in the first Tokyo Ghoul season?
5114|Can you name every main character listed for Fullmetal Alchemist: Brotherhood?
20|Who belongs to the original Naruto anime's main cast, excluding supporting roles?
20954|What are the names of A Silent Voice's main characters?
21519|Who are all the main characters in Your Name, according to the catalog?
127230|Which characters have main billing in Chainsaw Man's first anime season?
101759|Can I get the main-character names for The Promised Neverland season one?
20755|Who are the main characters in the first Assassination Classroom season?
21507|List everyone assigned a main character role in Mob Psycho 100 season one.
21355|Who are the main cast characters listed for Re:ZERO's first TV season?
20665|Which characters make up Your Lie in April's main cast?
1735|What is the full list of main characters for Naruto: Shippuden?
21234|Who has a main character role in ERASED?
9253|Please name the characters marked MAIN for the original Steins;Gate anime.
20464|Who is included in Haikyu season one's main character lineup?
4224|Which characters are listed as main rather than supporting in Toradora!?
30|Who are all the main characters credited for Neon Genesis Evangelion?
154587|Can you give the complete catalog main cast for Frieren: Beyond Journey's End?''',
}


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    rows, evidence = [], []
    with sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True) as db:
        for field, block in PROMPTS.items():
            for line in block.splitlines():
                anime_id, question = line.split('|', 1)
                record = json.loads(db.execute('SELECT data FROM anime WHERE id=?', (anime_id,)).fetchone()[0])
                values = database_values(db, anime_id, field)
                rows.append(dict(question_id=f'hf_{len(rows) + 1:03d}', anime_id=anime_id,
                                 title=record['title'], field=field, question=question,
                                 expected_values=json.dumps(values, ensure_ascii=False),
                                 question_origin='authored_naturalistic',
                                 answer_source='local_anilist_catalog',
                                 benchmark_version='1'))
                evidence.append({'question_id': rows[-1]['question_id'], 'anime_id': anime_id,
                                 'field': field, 'values': values})
        catalog_meta = dict(db.execute('SELECT key,value FROM meta'))
    assert len(rows) == 180
    assert len({r['question'].casefold() for r in rows}) == len(rows)
    assert len({(r['anime_id'], r['field']) for r in rows}) == len(rows)
    path = DEST / 'questions.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    load_questions(path, DATABASE)
    manifest = {
        'benchmark_version': '1', 'question_count': len(rows),
        'unique_anime': len({r['anime_id'] for r in rows}),
        'by_field': dict(Counter(r['field'] for r in rows)),
        'question_origin': 'Individually authored naturalistic English prompts; not real user logs or externally sourced questions.',
        'database_path': DATABASE.relative_to(ROOT).as_posix(),
        'catalog_meta': catalog_meta,
        'questions_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'definitions': {
            'year': 'First release year of the specific catalog entry; not a full calendar date.',
            'duration': 'Catalog minutes per episode; no opening/ending subtraction.',
            'main_animation_studios': 'All studios with main=true AND animation=true.',
            'main_characters': 'All appearances with role MAIN; supporting roles excluded.',
        },
    }
    (DEST / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('question_count', 'unique_anime', 'by_field')}, indent=2))


if __name__ == '__main__':
    build()
