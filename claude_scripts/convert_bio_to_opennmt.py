"""
Convert BIO-labeled JSONL splits to OpenNMT-py parallel text files.

  Source (.src): plain whitespace-tokenized sentence (unchanged)
  Target (.tgt): same sentence with <gender> and </gender> inserted as
                 separate space-delimited tokens around GENDER-labeled spans

Because <gender> and </gender> are written as regular whitespace-separated
tokens, OpenNMT-py's default (no-transforms) tokenizer picks them up as
vocabulary items without any special configuration.

Example
-------
  tokens : ["Die", "Lehrer:innen", "kommen", "morgen"]
  labels : ["O",   "B-GENDER",    "O",      "O"]

  src    : Die Lehrer:innen kommen morgen
  tgt    : Die <gender> Lehrer:innen </gender> kommen morgen

Negative examples (all-O labels) copy src → tgt unchanged so the model
learns to pass through sentences that contain no gender terms.

Usage
-----
  python convert_bio_to_opennmt.py <jsonl_dir> <output_dir>
  python convert_bio_to_opennmt.py data_merged/ opennmt_data/
"""

import os
import json
import argparse


OPEN_TAG  = '<gender>'
CLOSE_TAG = '</gender>'


def bio_to_src_tgt(tokens, labels):
    """Return (src_line, tgt_line) for one example.

    Handles edge cases:
    - Lone I-GENDER without a preceding B-GENDER: treated as start of span.
    - Consecutive B-GENDER tokens (two separate spans with no gap): each
      gets its own <gender>...</gender> pair.
    """
    src = ' '.join(tokens)

    tgt_parts = []
    in_span = False

    for tok, label in zip(tokens, labels):
        if label == 'B-GENDER':
            if in_span:                        # close previous span first
                tgt_parts.append(CLOSE_TAG)
            tgt_parts.append(OPEN_TAG)
            tgt_parts.append(tok)
            in_span = True
        elif label == 'I-GENDER':
            if not in_span:                    # defensive: lone I without B
                tgt_parts.append(OPEN_TAG)
                in_span = True
            tgt_parts.append(tok)
        else:                                  # O
            if in_span:
                tgt_parts.append(CLOSE_TAG)
                in_span = False
            tgt_parts.append(tok)

    if in_span:
        tgt_parts.append(CLOSE_TAG)

    tgt = ' '.join(tgt_parts)
    return src, tgt


def convert_split(jsonl_path, src_path, tgt_path):
    """Convert one JSONL split file to a pair of OpenNMT plain-text files."""
    n_positive = 0
    n_negative = 0

    with open(jsonl_path,  'r', encoding='utf-8') as jf, \
         open(src_path,    'w', encoding='utf-8') as sf, \
         open(tgt_path,    'w', encoding='utf-8') as tf:

        for line in jf:
            ex = json.loads(line)
            tokens = ex['tokens']
            labels = ex['labels']

            if not tokens:
                continue

            src, tgt = bio_to_src_tgt(tokens, labels)
            sf.write(src + '\n')
            tf.write(tgt + '\n')

            if any(l != 'O' for l in labels):
                n_positive += 1
            else:
                n_negative += 1

    return n_positive, n_negative


def main():
    parser = argparse.ArgumentParser(
        description='Convert BIO JSONL splits to OpenNMT-py src/tgt files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('jsonl_dir',  help='Directory with train/valid/test.jsonl')
    parser.add_argument('output_dir', help='Directory to write .src and .tgt files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        jsonl = os.path.join(args.jsonl_dir,  f'{split}.jsonl')
        src   = os.path.join(args.output_dir, f'{split}.src')
        tgt   = os.path.join(args.output_dir, f'{split}.tgt')

        if not os.path.isfile(jsonl):
            print(f"  Skipping {split} (not found: {jsonl})")
            continue

        n_pos, n_neg = convert_split(jsonl, src, tgt)
        total = n_pos + n_neg
        print(f"  {split:6s}  {total:7,} lines  "
              f"({n_pos:,} with gender tags, {n_neg:,} copy-through)  → {src}")


if __name__ == '__main__':
    main()
