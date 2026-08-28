"""
Internal helper: merge per-source JSONL splits and shuffle them.

Called by run_corpora_pipeline.sh.  Not intended for direct use.

For each split (train / valid / test):
  1. Read all lines from every source's <split>.jsonl.
  2. Shuffle deterministically with --seed.
  3. Write the combined file to --output-dir/<split>.jsonl.

This preserves equal representation: each source already has the same
--max-positive cap applied, so after merging no single source dominates.
"""

import os
import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description='Merge and shuffle per-source JSONL splits.')
    parser.add_argument('--source-dirs', nargs='+', required=True,
                        help='Per-source data directories (each has train/valid/test.jsonl)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write merged JSONL files')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        lines = []
        for src_dir in args.source_dirs:
            path = os.path.join(src_dir, f'{split}.jsonl')
            if not os.path.isfile(path):
                print(f"  Warning: {path} not found, skipping.")
                continue
            with open(path, 'r', encoding='utf-8') as f:
                src_lines = f.readlines()
            lines.extend(src_lines)
            print(f"  [{split}] {os.path.basename(src_dir):30s} {len(src_lines):6,} examples")

        random.shuffle(lines)
        out_path = os.path.join(args.output_dir, f'{split}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  [{split}] merged → {out_path}  ({len(lines):,} total)")


if __name__ == '__main__':
    main()
