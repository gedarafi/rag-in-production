"""Evaluate the anime recommendations against the AniList catalogue.

Asks the agent, and the same model without tools as a baseline, three kinds of
recommendation questions whose answers can be checked in the SQLite catalogue:
  explicit  "Recommend 5 <genre or tag> anime."             title must have that genre or tag
  vague     a tag description instead of the tag name        title must have that tag
  seed      "What should I watch after <title>?"             title must exist and differ from the seed
Every answer is also checked for invented titles (not found in the catalogue).

Run from the repository root after the AniList setup has finished:
    python evaluate_recommendations.py
The model comes from settings.toml, or from RAG_LLM_MODEL (e.g. RAG_LLM_MODEL=gemma3:4b).
Writes one row per answer to data/evaluation/recommendations_<model>.csv and prints a summary.
"""

import csv
import json
import random
import re
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path

from src.anime_tools import normalize
from src.config import load_settings
from src.rag import RagPipeline

QUESTIONS_PER_KIND = 30
RECOMMENDATION_COUNT = 5
MINIMUM_TITLES_PER_TERM = 30
SEED = 7
OUTPUT_FOLDER = Path("data/evaluation")
BASELINE_INSTRUCTION = "Answer with a numbered list of anime titles only, one title per line."


def sample_questions(database):
    random_generator = random.Random(SEED)
    terms = database.execute(
        "SELECT term, MIN(label) FROM features GROUP BY term HAVING COUNT(*) >= ?", (MINIMUM_TITLES_PER_TERM,)).fetchall()
    described_tags = database.execute(
        "SELECT f.term, i.description FROM features f JOIN entity_feature_info i ON i.id = f.term "
        "GROUP BY f.term HAVING COUNT(*) >= ?", (MINIMUM_TITLES_PER_TERM,)).fetchall()
    seeds = database.execute(
        "SELECT id, title FROM anime WHERE id IN (SELECT anime_id FROM recommendations GROUP BY anime_id "
        "HAVING COUNT(*) >= ?) ORDER BY popularity DESC LIMIT 300", (RECOMMENDATION_COUNT,)).fetchall()

    questions = []
    for term, label in random_generator.sample(terms, QUESTIONS_PER_KIND):
        questions.append(("explicit", f"Recommend {RECOMMENDATION_COUNT} {label} anime.", term))
    for term, description in random_generator.sample(described_tags, QUESTIONS_PER_KIND):
        questions.append(("vague", f"Recommend {RECOMMENDATION_COUNT} anime that match this description: {description}", term))
    for anime_id, title in random_generator.sample(seeds, QUESTIONS_PER_KIND):
        questions.append(("seed", f"What should I watch after {title}?", anime_id))
    return questions


def listed_titles(answer):
    """Titles from the numbered lines of an answer; bold text wins over the rest of the line.

    Bulleted lines are ignored: models use them for details under a title ("* **Genres:** ..."),
    which would otherwise be counted as invented titles.
    """
    titles = []
    for line in answer.splitlines():
        item = re.match(r"\s*\d+[.)]\s+(.+)", line)
        if not item:
            continue
        bold = re.search(r"\*\*(.+?)\*\*", item.group(1))
        title = bold.group(1) if bold else re.split(r"\s[-–—(]|:\s", item.group(1))[0]
        titles.append(title.strip(" *_\"'"))
    return titles


def catalogue_id(database, title):
    row = database.execute("SELECT anime_id FROM aliases WHERE alias = ?", (normalize(title),)).fetchone()
    return row[0] if row else None


def score(database, kind, target, titles):
    identifiers = [catalogue_id(database, title) for title in titles]
    found = [identifier for identifier in identifiers if identifier]
    if kind == "seed":
        matching = [identifier for identifier in found if identifier != target]
    else:
        matching = [identifier for identifier in found if database.execute(
            "SELECT 1 FROM features WHERE anime_id = ? AND term = ?", (identifier, target)).fetchone()]
    return {"listed": len(titles), "found": len(found), "invented": len(titles) - len(found), "matching": len(matching)}


def summarise(rows):
    print(f"{'kind':<9}{'system':<10}{'answers':>8}{'5 titles':>10}{'invented':>10}{'matching':>10}{'seconds (median)':>18}")
    for kind in ("explicit", "vague", "seed"):
        for system in ("agent", "baseline"):
            group = [row for row in rows if row["kind"] == kind and row["system"] == system]
            listed = sum(row["listed"] for row in group) or 1
            print(f"{kind:<9}{system:<10}{len(group):>8}"
                  f"{sum(row['listed'] == RECOMMENDATION_COUNT for row in group) / len(group):>10.0%}"
                  f"{sum(row['invented'] for row in group) / listed:>10.0%}"
                  f"{sum(row['matching'] for row in group) / listed:>10.0%}"
                  f"{statistics.median(row['seconds'] for row in group):>18.1f}")


def main():
    settings = load_settings("settings.toml", "anilist")
    pipeline = RagPipeline(settings)
    database = sqlite3.connect(settings.documents / "catalog.sqlite3")
    questions = sample_questions(database)
    # One file per model, so that runs with different RAG_LLM_MODEL values do not overwrite each other.
    output = OUTPUT_FOLDER / f"recommendations_{settings.llm.replace(':', '-')}.csv"
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Model: {settings.llm}. Writing {output}")

    fields = ["kind", "system", "question", "target", "seconds", "listed", "found", "invented", "matching",
              "seed_recognised", "answer"]
    # Resume: keep the questions whose two answers are already saved, and ask only the others.
    rows = []
    if output.exists():
        with output.open(encoding="utf-8", newline="") as file:
            saved = list(csv.DictReader(file))
        answers_per_question = Counter(row["question"] for row in saved)
        rows = [{**row, "seconds": float(row["seconds"]),
                 **{key: int(row[key]) for key in ("listed", "found", "invented", "matching")}}
                for row in saved if answers_per_question[row["question"]] == 2]
    finished = {row["question"] for row in rows}
    print(f"Resuming with {len(finished)} finished questions" if finished else "Starting a new run")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        for number, (kind, question, target) in enumerate(questions, start=1):
            if question in finished:
                continue
            for system in ("agent", "baseline"):
                started = time.perf_counter()
                try:
                    if system == "agent":
                        answer = pipeline.ask(question)
                    else:
                        answer = pipeline.models.answer(f"{question}\n{BASELINE_INSTRUCTION}",
                                                        settings.temperature, settings.max_tokens)
                except Exception as error:  # one failed answer must not stop a long run
                    answer = f"ERROR: {type(error).__name__}: {error}"
                seconds = round(time.perf_counter() - started, 2)
                # The plan, tool arguments and selected context of the last agent call, as text.
                trace = json.dumps(getattr(pipeline, "last_route", {}), default=str)
                seed_recognised = f'"{target}"' in trace if system == "agent" and kind == "seed" else ""
                row = {"kind": kind, "system": system, "question": question, "target": target, "seconds": seconds,
                       **score(database, kind, target, listed_titles(answer)),
                       "seed_recognised": seed_recognised, "answer": answer}
                writer.writerow(row)
                file.flush()
                rows.append(row)
            print(f"{number}/{len(questions)} {kind}: {question[:70]}")
    summarise(rows)


if __name__ == "__main__":
    main()
