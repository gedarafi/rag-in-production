"""Show how far the Gemma 3 4B and llama3.2:3b evaluation runs have got.

Reads the saved files only, so it is safe to run at any time while the evaluations are running.
Run from the repository root, in a second terminal:
    python evaluation_progress.py
"""

import csv
from pathlib import Path

NOT_QUESTIONS = {"manifest", "summary", "metrics", "answers"}


def checkpoints(folder):
    """Aufa's evaluations save one <question_id>.json per finished question."""
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.glob("*.json") if path.stem not in NOT_QUESTIONS)


def baseline_answers(path):
    """A question of evaluate_recommendations.py is finished once its baseline answer is saved."""
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as file:
        return sum(row["system"] == "baseline" for row in csv.DictReader(file))


for model, folder_name, file_name in (("Gemma 3 4B", "gemma3_4b", "gemma3-4b"), ("llama3.2:3b", "llama3_2_3b", "llama3.2-3b")):
    steps = [
        ("hard facts (Aufa)", checkpoints(f"data/evaluation/hard_facts/results/{folder_name}"), 180),
        ("recommendations (Aufa)", checkpoints(f"data/evaluation/recommendations/results/{folder_name}"), 72),
        ("baseline evaluation (mine)", baseline_answers(f"data/evaluation/recommendations_{file_name}.csv"), 90),
    ]
    print(model)
    for name, done, total in steps:
        print(f"  {name:<28}{done:>4}/{total}  {done / total:>4.0%}")
    done = sum(step[1] for step in steps)
    total = sum(step[2] for step in steps)
    print(f"  {'overall':<28}{done:>4}/{total}  {done / total:>4.0%}")
