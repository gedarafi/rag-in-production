"""Export recommendation benchmark coverage and actual saved-result charts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .recommendation_evaluation import load_dataset, summarize


def render(dataset, output, results=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str((output / '.matplotlib').resolve()))
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    questions, _, dataset_manifest = load_dataset(dataset)
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    artifacts = []

    def save(fig, name, caption):
        fig.tight_layout()
        for extension in ('png', 'svg'):
            fig.savefig(output / f'{name}.{extension}', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        artifacts.append({'png': str((output / f'{name}.png').resolve()),
                          'svg': str((output / f'{name}.svg').resolve()), 'caption': caption})

    categories = list(dict.fromkeys(q['category'] for q in questions))
    fig, ax = plt.subplots(figsize=(10, 5))
    left = np.zeros(len(categories))
    for wording, color in [('explicit', '#386cb0'), ('indirect', '#21a18c')]:
        counts = [sum(q['category'] == c and q['wording'] == wording for q in questions) for c in categories]
        ax.barh([c.replace('_', ' ').title() for c in categories], counts, left=left, label=wording.title(), color=color)
        left += counts
    ax.set(xlabel='Authored questions', title=f'Dataset coverage: {len(questions)} requests, {len(categories)} categories')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()
    ax.legend()
    save(fig, 'dataset_coverage', 'Complete dataset. Indirect wording describes a preference without naming every expected catalog label.')

    if results is not None:
        results = Path(results)
        rows = json.loads((results / 'answers.json').read_text(encoding='utf-8'))
        run_manifest = json.loads((results / 'manifest.json').read_text(encoding='utf-8'))
        if run_manifest['dataset_files'] != dataset_manifest['files_sha256']:
            raise ValueError('Saved run used a different frozen dataset')
        lookup = {q['question_id']: q for q in questions}
        expected_ids = run_manifest['question_ids']
        if len({r['question_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate saved question IDs')
        if any(r['question_id'] not in expected_ids or any(r.get(k) != v for k, v in lookup[r['question_id']].items()) for r in rows):
            raise ValueError('Results do not match the frozen dataset')
        if rows:
            note = f'{len(rows)}/{len(expected_ids)} selected requests attempted; {len(questions)} in full benchmark'
            if len(rows) < len(questions):
                note += ' (partial benchmark)'
            present = [c for c in categories if any(r['category'] == c for r in rows)]
            groups = [[r for r in rows if r['category'] == c] for c in present]
            stats = [summarize(group, len(group)) for group in groups]
            labels = [f'{c.replace("_", " ").title()} (n={len(g)})' for c, g in zip(present, groups)]
            y = np.arange(len(present))
            fig, ax = plt.subplots(figsize=(11, max(4, len(present) * .48)))
            ax.barh(y - .18, [100 * s['end_to_end_precision_at_k'] for s in stats], .36, color='#21918c', label='End-to-end strict P@k')
            ax.barh(y + .18, [100 * s['scoring_coverage'] for s in stats], .36, color='#9dbde0', label='Scoring coverage')
            ax.set(yticks=y, yticklabels=labels, xlim=(0, 105), xlabel='Percent', title='Strict recommendation compliance by request type\n' + note)
            ax.invert_yaxis()
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9)
            save(fig, 'compliance_by_category', 'Each recommendation must meet every constraint. Missing slots, unresolved titles, and duplicates score zero. Errors remain in the end-to-end denominator.')

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            for ax, axis in zip(axes, ('wording', 'category')):
                labels2 = sorted({r[axis] for r in rows})
                x = np.arange(len(labels2))
                for offset, metric, color in [(-.24, 'precision_at_k', '#386cb0'), (0, 'graded_fit_at_k', '#21918c'), (.24, 'ndcg_at_k', '#e9a23b')]:
                    values = []
                    for label in labels2:
                        values2 = [r['metrics'][metric] for r in rows if r[axis] == label and r['metrics'] is not None]
                        values.append(100 * sum(values2) / len(values2) if values2 else float('nan'))
                    ax.bar(x + offset, values, .24, color=color, label=metric.replace('_', ' '))
                for i, label in enumerate(labels2):
                    if not any(r[axis] == label and r['metrics'] is not None for r in rows):
                        ax.text(i, 3, 'Unscored', ha='center', rotation=90, color='#777777', fontsize=8)
                ax.set(xticks=x, xticklabels=[c.replace('_', ' ') for c in labels2], ylim=(0, 105), ylabel='Percent', title=axis.title() + ' (scored requests only)')
                ax.tick_params(axis='x', labelrotation=60 if axis == 'category' else 0)
            axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.15), fontsize=8)
            fig.suptitle('Strict, graded, and ranking metrics\n' + note, fontsize=11)
            save(fig, 'quality_breakdowns', 'Scored-answer macro averages. Missing scores have no bar. nDCG uses the best possible unique catalog matches for each specific request, not a fixed hand-picked list.')

            names = ['scored', 'unanswered', 'needs_review', 'extraction_error', 'model_error']
            colors = ['#21918c', '#999999', '#dba441', '#9152ad', '#cf4446']
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            counts = Counter(r['status'] for r in rows)
            axes[0].barh(names, [counts[n] for n in names], color=colors)
            axes[0].set(xlabel='Requests', title='Answer and extraction outcomes')
            resolved = Counter(d['resolution'] for r in rows for d in r['recommendations'])
            axes[1].barh(['resolved', 'ambiguous', 'unknown'], [resolved[n] for n in ('resolved', 'ambiguous', 'unknown')], color=['#21918c', '#dba441', '#cf4446'])
            axes[1].set(xlabel='Extracted recommendations (all positions)', title='Title resolution audit')
            fig.suptitle(note, fontsize=10)
            save(fig, 'outcomes_and_resolution', 'LLM extraction is fallible: uncertain parses are held for review. Ambiguous or out-of-catalog names never receive a guessed title identity.')

            detail = [(r, d) for r in rows for d in r['recommendations'] if d['rank'] <= r['k']][:30]
            if detail:
                signatures = sorted({json.dumps(c, sort_keys=True) for r, _ in detail for c in r['constraints']})
                constraints = [json.loads(s) for s in signatures]
                matrix = np.full((len(detail), len(constraints)), np.nan)
                row_labels = []
                for i, (r, d) in enumerate(detail):
                    row_labels.append(f'{r["question_id"]} #{d["rank"]}: {d["extracted_title"][:30]}')
                    for z, c in enumerate(r['constraints']):
                        matrix[i, signatures.index(json.dumps(c, sort_keys=True))] = d['constraint_scores'][z]
                column_labels = []
                for c in constraints:
                    value = ' / '.join(c['value']) if isinstance(c['value'], list) else str(c['value'])
                    field = c['field'].replace('_', ' ').title()
                    column_labels.append(f'{field}\n{c["op"]}: {value}')
                fig, ax = plt.subplots(figsize=(max(13, len(constraints) * .8), max(4, len(detail) * .3)))
                cmap = plt.get_cmap('YlGnBu').copy()
                cmap.set_bad('#eeeeee')
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect='auto')
                ax.set(xticks=np.arange(len(constraints)), xticklabels=column_labels, yticks=np.arange(len(detail)), yticklabels=row_labels,
                       title='Per-recommendation constraint fit (first 30 returned top-k items)\n' + note)
                plt.setp(ax.get_xticklabels(), rotation=50, ha='right', fontsize=8)
                fig.colorbar(im, ax=ax, label='Constraint fit: 0 to 1')
                save(fig, 'recommendation_constraint_fit', 'Each column is an individual expected constraint, including each required or excluded tag group. Gray means it was not requested for that item. Duplicate flags and identities are in recommendations.csv. Missing slots are excluded from this audit heatmap but penalized in P@k.')

            scored = [r for r in rows if r['metrics'] is not None]
            if scored:
                fig, ax = plt.subplots(figsize=(11, max(4, len(scored) * .34)))
                y = np.arange(len(scored))
                literal = [100 * r['metrics']['literal_precision_at_k'] for r in scored]
                rescued = [100 * r['metrics']['semantic_credit_at_k'] for r in scored]
                ax.barh(y, literal, color='#386cb0', label='Literal credit')
                ax.barh(y, rescued, left=literal, color='#21a18c', label='Accepted implication credit')
                ax.set(yticks=y, yticklabels=[r['question_id'] for r in scored], xlim=(0, 105), xlabel='Percent of requested slots',
                       title='Credit from literal and reviewed alternative evidence\n' + note)
                ax.invert_yaxis()
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9)
                save(fig, 'literal_and_semantic_credit', 'The stacked total equals intent-aware strict P@k. No extra labels are penalized. All-explicit samples may have no implication credit; alternative-path details are saved in the recommendation audit.')

            fig, ax = plt.subplots(figsize=(10, 4))
            scored = [r for r in rows if r['metrics'] is not None]
            if scored:
                for metric, label, color in [('model_seconds', 'RAG answer', '#386cb0'), ('extraction_seconds', 'Title extraction', '#e9a23b')]:
                    values = [r[metric] for r in scored]
                    ax.scatter(range(1, len(scored) + 1), values, label=label, color=color)
                ax.set(xlabel='Scored request in execution order', ylabel='Seconds', title='Generation and LLM extraction latency\n' + note)
                ax.legend()
                save(fig, 'latency', 'Actual measured timings; cold starts and routing are included. Failures are reported in the outcome chart.')
            else:
                plt.close(fig)
    (output / 'charts.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/evaluation/recommendations')
    parser.add_argument('--output', required=True)
    parser.add_argument('--results')
    args = parser.parse_args()
    render(args.dataset, args.output, args.results)


if __name__ == '__main__':
    main()
