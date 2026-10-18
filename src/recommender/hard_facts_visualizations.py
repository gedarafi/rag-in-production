"""Render dataset and saved hard-fact evaluation charts as PNG and SVG artifacts."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .factual_evaluation import read_csv

LABELS = {
    'year': 'Release year', 'episodes': 'Episode count', 'duration': 'Episode runtime',
    'format': 'Format', 'source_material': 'Source material',
    'main_animation_studios': 'Animation studios', 'main_characters': 'Main characters',
}


def render(questions_csv, output, results=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((output / '.matplotlib').resolve()))
    # Jupyter exports its inline backend to children, but this environment does not
    # need (or necessarily include) matplotlib_inline. Select the file backend
    # before importing matplotlib: setting it afterwards is already too late.
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    artifacts = []

    def save(fig, name, caption):
        fig.tight_layout()
        for extension in ('png', 'svg'):
            fig.savefig(output / f'{name}.{extension}', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        artifacts.append({'png': str((output / f'{name}.png').resolve()),
                          'svg': str((output / f'{name}.svg').resolve()), 'caption': caption})

    questions = read_csv(questions_csv)
    fields = [f for f in LABELS if any(q['field'] == f for q in questions)]
    counts = Counter(q['field'] for q in questions)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([LABELS[f] for f in fields], [counts[f] for f in fields], color='#386cb0')
    for i, field in enumerate(fields):
        ax.text(counts[field] + .2, i, str(counts[field]), va='center')
    ax.set(xlabel='Questions', title=f'Dataset coverage: {len(questions)} questions, {len({q["anime_id"] for q in questions})} anime',
           xlim=(0, max(counts.values()) * 1.16))
    ax.invert_yaxis()
    save(fig, 'dataset_coverage', 'Full CSV coverage; independent of how many evaluation questions have been attempted.')
    if results is None:
        (output / 'charts.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
        return artifacts

    rows = read_csv(Path(results) / 'answers.csv')
    if not rows:
        (output / 'charts.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
        return artifacts
    # Derive outcomes directly from saved answer rows, rather than substituting missing scores with success.
    row_ids = [r['question_id'] for r in rows]
    if len(set(row_ids)) != len(rows) or not set(row_ids) <= {q['question_id'] for q in questions}:
        raise ValueError('Result question IDs must be unique and belong to the supplied benchmark.')
    benchmark = {q['question_id']: q for q in questions}
    for row in rows:
        if any(row[key] != benchmark[row['question_id']][key] for key in ('question', 'anime_id', 'field', 'expected_values')):
            raise ValueError('Saved results do not match the supplied benchmark.')
    fields = [f for f in LABELS if any(r['field'] == f for r in rows)]
    groups = {f: [r for r in rows if r['field'] == f] for f in fields}
    labels = [f'{LABELS[f]} (n={len(groups[f])})' for f in fields]
    note = f'{len(rows)}/{len(questions)} benchmark questions attempted'
    if len(rows) < len(questions):
        note += ' — partial run; not a full benchmark score'
    x = np.arange(len(fields))

    fig, ax = plt.subplots(figsize=(11, 5))
    exact = [100 * sum(r['exact_match'] == '1' for r in groups[f]) / len(groups[f]) for f in fields]
    coverage = [100 * sum(r['exact_match'] != '' for r in groups[f]) / len(groups[f]) for f in fields]
    ax.barh(x - .18, exact, .36, label='Exact accuracy / all attempts', color='#21918c')
    ax.barh(x + .18, coverage, .36, label='Automatically scored / all attempts', color='#9dbde0')
    ax.set(yticks=x, yticklabels=labels, xlim=(0, 110), xlabel='Percent', title='Accuracy and scoring coverage by fact type\n' + note)
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, fontsize=8)
    save(fig, 'accuracy_by_field', 'Errors remain in the accuracy denominator. Missing fields are omitted, not displayed as zero performance.')

    outcome_names = ['Correct', 'Incorrect', 'Unanswered', 'Model error', 'Extraction error', 'Needs review']
    colors = ['#21918c', '#e9a23b', '#a3a3a3', '#cf4446', '#9152ad', '#6b83a5']

    def outcome(row):
        if row['status'] in {'model_error', 'extraction_error', 'needs_review', 'unanswered'}:
            return {'model_error': 'Model error', 'extraction_error': 'Extraction error',
                    'needs_review': 'Needs review', 'unanswered': 'Unanswered'}[row['status']]
        return 'Correct' if row['exact_match'] == '1' else 'Incorrect'

    fig, ax = plt.subplots(figsize=(11, 5))
    left = np.zeros(len(fields))
    for name, color in zip(outcome_names, colors):
        values = np.array([sum(outcome(r) == name for r in groups[f]) for f in fields])
        if values.sum():
            ax.barh(x, values, left=left, color=color, label=name)
            left += values
    ax.set(yticks=x, yticklabels=labels, xlabel='Attempted questions', title='Answer outcomes by fact type\n' + note)
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3, fontsize=8)
    save(fig, 'answer_outcomes', 'Counts distinguish factual mismatches from inference failures and uncertain extraction.')

    fig, ax = plt.subplots(figsize=(11, 5))
    scored_counts = [sum(r['exact_match'] != '' for r in groups[f]) for f in fields]
    for offset, metric, color in [(-.25, 'precision', '#386cb0'), (0, 'recall', '#21918c'), (.25, 'f1', '#e9a23b')]:
        values = []
        for f in fields:
            scored = [float(r[metric]) for r in groups[f] if r[metric] != '']
            values.append(100 * sum(scored) / len(scored) if scored else float('nan'))
        ax.bar(x + offset, values, .25, label=metric.upper() if metric == 'f1' else metric.title(), color=color)
    for i, count in enumerate(scored_counts):
        if count == 0:
            ax.text(i, 4, 'Unscored', ha='center', rotation=90, color='#777777')
    ax.set(xticks=x, xticklabels=[f'{LABELS[f]}\nscored n={n}' for f, n in zip(fields, scored_counts)],
           ylim=(0, 110), ylabel='Percent', title='Macro value metrics among scored answers\n' + note)
    ax.tick_params(axis='x', labelrotation=25)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    save(fig, 'value_metrics', 'Precision penalizes extra values; recall penalizes missing values. Unscored groups have no bar, not a zero score.')

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, key, color, label in [(-.18, 'model_seconds', '#386cb0', 'RAG response'),
                                       (.18, 'extraction_seconds', '#e9a23b', 'Fact extraction')]:
        means = [sum(float(r[key]) for r in groups[f]) / len(groups[f]) for f in fields]
        ax.barh(x + offset, means, .36, color=color, label=label)
    ax.set(yticks=x, yticklabels=labels, xlabel='Mean seconds per attempted question', title='Response and extraction latency\n' + note)
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, fontsize=8)
    save(fig, 'latency_by_field', 'All attempts included; extraction is zero when generation failed. First-response latency includes model startup.')

    (output / 'charts.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--results')
    args = parser.parse_args()
    render(args.questions, args.output, args.results)


if __name__ == '__main__':
    main()
