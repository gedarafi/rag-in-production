"""Re-score the saved recommendation answers and print the results table.

Reads every data/evaluation/recommendations_<model>.csv written by evaluate_recommendations.py,
extracts the titles again with the current rules (no model is called) and prints one table per
model, followed by the same numbers as LaTeX table rows for the report.

Run from the repository root:
    python summarise_recommendations.py
"""

import csv
import sqlite3
import statistics
from pathlib import Path

from evaluate_recommendations import OUTPUT_FOLDER, RECOMMENDATION_COUNT, listed_titles, score

DATABASE = Path("data/documents/anilist/catalog.sqlite3")


def main():
    database = sqlite3.connect(DATABASE)
    latex_rows = []
    for path in sorted(OUTPUT_FOLDER.glob("recommendations_*.csv")):
        model = path.stem.removeprefix("recommendations_")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        print(f"\n{model}")
        print(f"{'kind':<9}{'system':<10}{'answers':>8}{'no list':>9}{'5 titles':>10}{'invented':>10}"
              f"{'precision':>11}{'seconds (median)':>18}")
        for kind in ("explicit", "vague", "seed"):
            for system in ("agent", "baseline"):
                group = [row for row in rows if row["kind"] == kind and row["system"] == system]
                scores = [score(database, kind, row["target"], listed_titles(row["answer"])) for row in group]
                listed = sum(item["listed"] for item in scores) or 1
                without_list = sum(item["listed"] == 0 for item in scores)
                five = sum(item["listed"] == RECOMMENDATION_COUNT for item in scores) / len(group)
                invented = sum(item["invented"] for item in scores) / listed
                precision = sum(item["matching"] for item in scores) / listed
                seconds = statistics.median(float(row["seconds"]) for row in group)
                print(f"{kind:<9}{system:<10}{len(group):>8}{without_list:>9}{five:>10.0%}{invented:>10.0%}"
                      f"{precision:>11.0%}{seconds:>18.1f}")
                latex_rows.append(f"{model} & {kind} & {system} & {five:.0%} & {invented:.0%} & {precision:.0%} "
                                  f"& {seconds:.1f} \\\\".replace("%", r"\%"))
    print("\nLaTeX rows: model & kind & system & five titles & hallucination & precision & seconds")
    print("\n".join(latex_rows))


if __name__ == "__main__":
    main()
