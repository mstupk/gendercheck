"""
Assemble the final Stage 2 training set (SPECIFICATION.md §3.2 / §8
milestone 3): Stage 1's original data (broad positive/negative coverage) +
07b_mine_stage2_hard_negatives.py's output (the targeted hard negatives and
bonus positives that make this "Stage 2" rather than just "more Stage 1
data").

Tags every example with a 'source' field (adding 'stage1' where missing --
07b's output is already tagged) so the eventual evaluation can isolate the
hard-negative subset specifically, which is the only way to measure whether
hard-negative mining actually worked rather than just eyeballing overall F1.

Usage:
    python 08_build_stage2_dataset.py \\
        --stage1-dir ../claude_pipeline_output_stage1/data_merged \\
        --hard-mined-dir ../claude_pipeline_output_stage2/hard_mined \\
        --output-dir ../claude_pipeline_output_stage2/data_merged \\
        --seed 42
"""

import os
import json
import random
import argparse


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage1-dir', required=True)
    parser.add_argument('--hard-mined-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--true-neg-cap-multiplier', type=float, default=3.0,
                        help='Cap stage2_true_neg examples per split to this many times '
                             'the stage2_hard_neg count in that split -- 07b mined far more '
                             'plain true negatives than hard negatives (a byproduct of how '
                             'rare Stage 1 false positives are, not a deliberate ratio), and '
                             'the whole point of Stage 2 is to not let easy negatives drown '
                             'out the hard-negative signal (see SPECIFICATION.md §3.2: '
                             '"not just random negatives ... too easy"). Set <=0 to disable.')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        stage1_examples = load_jsonl(os.path.join(args.stage1_dir, f'{split}.jsonl'))
        for ex in stage1_examples:
            ex.setdefault('source', 'stage1')

        hard_mined_examples = load_jsonl(os.path.join(args.hard_mined_dir, f'{split}.jsonl'))

        if args.true_neg_cap_multiplier > 0:
            hard_neg_n = sum(1 for ex in hard_mined_examples if ex.get('source') == 'stage2_hard_neg')
            cap = max(1, int(hard_neg_n * args.true_neg_cap_multiplier))
            true_neg = [ex for ex in hard_mined_examples if ex.get('source') == 'stage2_true_neg']
            other = [ex for ex in hard_mined_examples if ex.get('source') != 'stage2_true_neg']
            random.shuffle(true_neg)
            dropped = max(0, len(true_neg) - cap)
            hard_mined_examples = other + true_neg[:cap]
            if dropped:
                print(f"[{split}] capping stage2_true_neg: {len(true_neg):,} -> {min(cap, len(true_neg)):,} "
                      f"(dropped {dropped:,}, cap = {args.true_neg_cap_multiplier}x hard_neg={hard_neg_n:,})")

        combined = stage1_examples + hard_mined_examples
        random.shuffle(combined)

        by_source = {}
        for ex in combined:
            by_source[ex['source']] = by_source.get(ex['source'], 0) + 1

        out_path = os.path.join(args.output_dir, f'{split}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in combined:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        breakdown = '  '.join(f'{k}={v:,}' for k, v in sorted(by_source.items()))
        print(f"[{split}] {len(combined):,} total ({breakdown}) -> {out_path}")


if __name__ == '__main__':
    main()
