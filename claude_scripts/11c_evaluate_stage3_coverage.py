"""
Measure the rule-based Stage 3 inflector's (11b_stage3_inflect.py) coverage
against real mined (stem, convention, form) pairs -- SPECIFICATION.md §8
milestone 4's explicit requirement: "only add a learned fallback once the
rule table's coverage gap is measured against held-out mined pairs."

Splits by STEM (not by individual record) into train/test so the same word
never appears on both sides -- mirrors the article-level splitting hygiene
used throughout this project, applied at the natural leakage unit here.
"Train" isn't actually used to fit anything (the rule has no free
parameters) -- it's reserved so a future learned fallback, if the coverage
gap warrants one, has a fair train set that was never looked at while
characterizing the gap.

Reports both:
  - type-level accuracy (each unique (stem, convention) pair counted once)
  - token-level / frequency-weighted accuracy (weighted by real corpus
    frequency -- the number that reflects how often the rule would actually
    be wrong in practice, since common words dominate real usage)

Usage:
    python 11c_evaluate_stage3_coverage.py \\
        --pairs ../claude_pipeline_output_stage3/inflection_pairs.json \\
        --out ../claude_pipeline_output_stage3/coverage_results.json
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import util as _ilu
_spec = _ilu.spec_from_file_location("stage3_inflect", os.path.join(os.path.dirname(os.path.abspath(__file__)), "11b_stage3_inflect.py"))
inflect_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(inflect_mod)


def rule_matches(stem, convention, core_form):
    for singular in (False, True):
        try:
            if inflect_mod.inflect(stem, convention, singular=singular) == core_form:
                return True
        except ValueError:
            return False
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--min-count', type=int, default=1,
                        help='Ignore records seen fewer than this many times (noise filter)')
    args = parser.parse_args()

    with open(args.pairs, 'r', encoding='utf-8') as f:
        records = json.load(f)
    records = [r for r in records if r['count'] >= args.min_count and r['convention'] != 'paired']
    print(f"Loaded {len(records):,} (stem, convention, form) records "
          f"(min_count={args.min_count}, excluding 'paired' -- see 11b's docstring).")

    stems = sorted({r['stem'] for r in records})
    random.seed(args.seed)
    random.shuffle(stems)
    n_train = int(len(stems) * args.train_ratio)
    train_stems = set(stems[:n_train])
    test_records = [r for r in records if r['stem'] not in train_stems]
    print(f"Stem-level split: {n_train:,} train stems / {len(stems) - n_train:,} test stems "
          f"-> {len(test_records):,} test records.")

    per_convention = defaultdict(lambda: {'type_hit': 0, 'type_total': 0, 'freq_hit': 0, 'freq_total': 0})
    failures = []

    for r in test_records:
        hit = rule_matches(r['stem'], r['convention'], r['core_form'])
        c = per_convention[r['convention']]
        c['type_total'] += 1
        c['type_hit'] += int(hit)
        c['freq_total'] += r['count']
        c['freq_hit'] += r['count'] if hit else 0
        if not hit:
            failures.append(r)

    failures.sort(key=lambda r: -r['count'])

    results = {'per_convention': {}, 'overall': {}}
    tot = {'type_hit': 0, 'type_total': 0, 'freq_hit': 0, 'freq_total': 0}
    for conv, c in sorted(per_convention.items()):
        type_acc = c['type_hit'] / c['type_total'] if c['type_total'] else 0
        freq_acc = c['freq_hit'] / c['freq_total'] if c['freq_total'] else 0
        results['per_convention'][conv] = {
            'type_accuracy': type_acc, 'type_n': c['type_total'],
            'freq_weighted_accuracy': freq_acc, 'freq_n': c['freq_total'],
        }
        print(f"  [{conv:10s}] type-level: {c['type_hit']:,}/{c['type_total']:,} ({type_acc:.1%})  "
              f"freq-weighted: {c['freq_hit']:,}/{c['freq_total']:,} ({freq_acc:.1%})")
        for k in tot:
            tot[k] += c[k]

    overall_type = tot['type_hit'] / tot['type_total'] if tot['type_total'] else 0
    overall_freq = tot['freq_hit'] / tot['freq_total'] if tot['freq_total'] else 0
    results['overall'] = {
        'type_accuracy': overall_type, 'type_n': tot['type_total'],
        'freq_weighted_accuracy': overall_freq, 'freq_n': tot['freq_total'],
    }
    print(f"\n  OVERALL type-level: {tot['type_hit']:,}/{tot['type_total']:,} ({overall_type:.1%})")
    print(f"  OVERALL freq-weighted: {tot['freq_hit']:,}/{tot['freq_total']:,} ({overall_freq:.1%})")

    print(f"\nTop {min(20, len(failures))} most frequent rule failures (stem, convention, rule-would-produce, actual):")
    for r in failures[:20]:
        try:
            rule_out = inflect_mod.inflect(r['stem'], r['convention'])
        except ValueError:
            rule_out = '<error>'
        print(f"  {r['stem']!r:25s} [{r['convention']:10s}] rule={rule_out!r:25s} actual={r['core_form']!r:25s} (n={r['count']})")

    results['top_failures'] = [
        {'stem': r['stem'], 'convention': r['convention'], 'actual_core_form': r['core_form'], 'count': r['count']}
        for r in failures[:50]
    ]

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
