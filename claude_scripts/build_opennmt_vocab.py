"""
Build OpenNMT-py compatible vocabulary files from plain-text corpora.

OpenNMT-py 3.x vocabulary format: one "word<TAB>count" entry per line,
sorted by descending frequency (matches what onmt_build_vocab produces).
Special tokens (<blank>, <unk>, <s>, </s>) are added at the top.

Usage
-----
  python build_opennmt_vocab.py --src train.src --tgt train.tgt \
      --src-vocab vocab/vocab.src --tgt-vocab vocab/vocab.tgt \
      --src-size 50000 --tgt-size 50000

  # To guarantee that specific tokens are always in tgt vocab:
  python build_opennmt_vocab.py ... --tgt-extra "<gender> </gender>"
"""

import os
import argparse
from collections import Counter


# OpenNMT-py 3.x default special tokens (always prepended in this order)
SPECIAL_TOKENS = ['<blank>', '<unk>', '<s>', '</s>']


def count_tokens(filepath):
    """Return a Counter of all whitespace-split tokens in a file."""
    counts = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            counts.update(line.split())
    return counts


def write_vocab(counts, special_extras, max_size, out_path):
    """Write a vocab file in OpenNMT-py format.

    Special tokens come first with a dummy count of 1, then regular tokens
    sorted by descending frequency, then any forced extras that weren't
    already present.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Remove special tokens from the normal counts to avoid duplication
    for tok in SPECIAL_TOKENS:
        counts.pop(tok, None)

    # Build ordered list: special tokens first, then by frequency
    entries = [(tok, 1) for tok in SPECIAL_TOKENS]
    entries += counts.most_common()

    # Trim to max_size (special tokens always included)
    if max_size > 0:
        entries = entries[:max_size]

    # Append forced extra tokens if they're not already present
    present = {tok for tok, _ in entries}
    for tok in special_extras:
        if tok and tok not in present:
            entries.append((tok, 1))

    with open(out_path, 'w', encoding='utf-8') as f:
        for tok, count in entries:
            f.write(f'{tok}\t{count}\n')

    return len(entries)


def main():
    parser = argparse.ArgumentParser(
        description='Build OpenNMT-py vocab files from plain-text corpora.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--src',       required=True, help='Source training file (.src)')
    parser.add_argument('--tgt',       required=True, help='Target training file (.tgt)')
    parser.add_argument('--src-vocab', required=True, help='Output src vocab path')
    parser.add_argument('--tgt-vocab', required=True, help='Output tgt vocab path')
    parser.add_argument('--src-size',  type=int, default=50000,
                        help='Max src vocabulary size (0 = unlimited)')
    parser.add_argument('--tgt-size',  type=int, default=50000,
                        help='Max tgt vocabulary size (0 = unlimited)')
    parser.add_argument('--tgt-extra', default='<gender> </gender>',
                        help='Space-separated tokens to force into tgt vocab')
    args = parser.parse_args()

    tgt_extras = args.tgt_extra.split() if args.tgt_extra else []

    print(f"Counting src tokens from {args.src} ...")
    src_counts = count_tokens(args.src)
    print(f"  {sum(src_counts.values()):,} tokens, {len(src_counts):,} unique")

    print(f"Counting tgt tokens from {args.tgt} ...")
    tgt_counts = count_tokens(args.tgt)
    print(f"  {sum(tgt_counts.values()):,} tokens, {len(tgt_counts):,} unique")

    n_src = write_vocab(src_counts, [],          args.src_size, args.src_vocab)
    n_tgt = write_vocab(tgt_counts, tgt_extras,  args.tgt_size, args.tgt_vocab)

    print(f"Wrote src vocab: {args.src_vocab}  ({n_src:,} entries)")
    print(f"Wrote tgt vocab: {args.tgt_vocab}  ({n_tgt:,} entries)")

    # Verify forced tokens made it in
    with open(args.tgt_vocab, 'r', encoding='utf-8') as f:
        tgt_vocab_tokens = {line.split('\t')[0] for line in f}
    for tok in tgt_extras:
        status = "✓" if tok in tgt_vocab_tokens else "MISSING!"
        print(f"  tgt forced token '{tok}': {status}")


if __name__ == '__main__':
    main()
