"""
Build the CONFIRMED masculine-stem vocabulary for Stage 2 hard-negative
mining (SPECIFICATION.md §3.2, milestone 3 in §8).

A "confirmed stem" is a word that was observed, somewhere in the corpus, as
the masculine base of a real mined gendered term (e.g. "Lehrer" is confirmed
because "Lehrer:innen" appears somewhere). This is the ground-truth oracle
07b_mine_stage2_hard_negatives.py uses to tell apart two kinds of Stage 1
model predictions on fresh text:
  - flagged a confirmed stem            -> a real candidate (bonus positive)
  - flagged something never confirmed   -> Stage 1 was fooled (hard negative)

This is a full, uncapped scan of every corpus source (unlike
01b_extract_masculine_spans.py's --max-positive-capped extraction), so the
resulting set is a strict superset of what Stage 1 actually trained on --
important so a real positive occurring outside Stage 1's 10,000-per-source
training cap doesn't get mislabeled as a hard negative.

It only needs to build a set of strings, not per-sentence training examples,
so it's much cheaper than a full extraction pass despite covering every file.

Usage:
    python 07a_build_confirmed_stems.py ../corpora/taz.de ../corpora/www.woz.ch \\
        --output ../claude_pipeline_output_stage2/confirmed_stems.json
"""

import os
import re
import json
import argparse
from bs4 import BeautifulSoup

# Identical to 01_extract_training_data.py / 01b_extract_masculine_spans.py --
# kept as a literal copy per this project's established convention (see those
# files' docstrings) so each extraction script stays a self-contained artifact.
_ALPHA = r'[a-zA-ZäöüÄÖÜß]'
_ALPHA_LOWER = r'[a-zäöüß]'
_ALPHA_UPPER = r'[A-ZÄÖÜ]'

GENDER_PATTERNS = {
    'paired':    re.compile(r'\b(' + _ALPHA + r'{3,})\s+und\s+\1(?:innen|Innen)\b'),
    'asterisk':  re.compile(r'\b' + _ALPHA_UPPER + r'[^\*\s]{2,}\*' + _ALPHA_LOWER + r'{2,}\b'),
    'colon':     re.compile(r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}:' + _ALPHA_LOWER + r'{2,}\b'),
    'underscore':re.compile(r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}_' + _ALPHA_LOWER + r'{2,}\b'),
    'binnen_i':  re.compile(r'\b' + _ALPHA_UPPER + _ALPHA_LOWER + r'{2,}I' + _ALPHA_LOWER + r'{2,}\b'),
}


def degender_match(match_obj, pattern_name):
    text = match_obj.group(0)
    if pattern_name == 'paired':
        return match_obj.group(1)
    if pattern_name == 'asterisk':
        return text.split('*', 1)[0]
    if pattern_name == 'colon':
        return text.split(':', 1)[0]
    if pattern_name == 'underscore':
        return text.split('_', 1)[0]
    if pattern_name == 'binnen_i':
        idx = text.index('I', 1)
        return text[:idx]
    raise ValueError(f"unknown pattern: {pattern_name}")


def is_html(path):
    try:
        with open(path, 'rb') as f:
            head = f.read(512).lower()
        return b'<!doctype' in head or b'<html' in head
    except OSError:
        return False


def stems_from_file(path, all_files):
    if not all_files and not path.lower().endswith('.html'):
        return set()
    if all_files and not path.lower().endswith('.html') and not is_html(path):
        return set()
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
    except Exception:
        return set()

    stems = set()
    for name, pattern in GENDER_PATTERNS.items():
        for m in pattern.finditer(text):
            stems.add(degender_match(m, name))
    return stems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('corpus_dirs', nargs='+', help='One or more corpus source directories')
    parser.add_argument('--output', required=True, help='Output JSON path (sorted list of stems)')
    parser.add_argument('--progress-every', type=int, default=5000)
    args = parser.parse_args()

    all_stems = set()
    for corpus_dir in args.corpus_dirs:
        source_name = os.path.basename(os.path.normpath(corpus_dir))
        # Always content-sniff non-.html-extension files (is_html()) rather than
        # relying on a directory-level heuristic -- correct and cheap for both
        # taz.de (all .html) and www.woz.ch (extensionless) with one code path.
        print(f"Scanning {corpus_dir} ...")
        n_files = 0
        for dirpath, _, filenames in os.walk(corpus_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                all_stems |= stems_from_file(path, all_files=True)
                n_files += 1
                if n_files % args.progress_every == 0:
                    print(f"  [{source_name}] {n_files:,} files scanned, "
                          f"{len(all_stems):,} confirmed stems so far ...")
        print(f"  [{source_name}] done: {n_files:,} files scanned.")

    print(f"\nTotal confirmed stems: {len(all_stems):,}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(sorted(all_stems), f, ensure_ascii=False, indent=1)
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
