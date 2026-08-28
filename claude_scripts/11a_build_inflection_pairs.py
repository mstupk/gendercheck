"""
Build the ground-truth (stem, convention, real gendered surface form)
dataset for Stage 3 (SPECIFICATION.md §3.2 / §8 milestone 4: case-aware
inflection). Full, uncapped corpus scan -- same regex set as every other
extraction script here -- but unlike 07a_build_confirmed_stems.py (which
only keeps the deduplicated stem), this keeps the FULL matched surface form
too, with a frequency count per (stem, convention, surface_form) triple.

This is the empirical basis for both:
  1. the rule-based generator (11b_stage3_inflect.py) -- read off the actual
     separator/suffix convention directly from real data rather than assumed
     from the regex definitions alone
  2. its coverage evaluation (11c_evaluate_stage3_coverage.py) -- held out a
     fraction of these real pairs and check how often the rule reproduces
     the real form exactly

Usage:
    python 11a_build_inflection_pairs.py ../corpora/taz.de ../corpora/www.woz.ch \\
        --output ../claude_pipeline_output_stage3/inflection_pairs.json
"""

import os
import re
import json
import argparse
from collections import Counter
from bs4 import BeautifulSoup

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


# German compounding means the regexes -- which only require ">=2 lowercase
# chars" after the separator, with no upper bound -- often swallow a
# following compound continuation as part of the match: "Nutzer:innenmenu"
# (user menu), "Richter:innenwahl" (judges' election). Confirmed empirically
# on a smoke sample: ~10% of unique (stem, form) pairs, and a much larger
# share of raw match VOLUME since these are often repeated site-chrome
# strings (nav labels, tag links). The suffix itself is always exactly
# "innen" (plural) or "in" (singular) -- so truncate the match right after
# that suffix to recover the clean "core" gendered form regardless of
# whatever compound tail follows.
_SUFFIX_CANDIDATES = ['innen', 'in']  # longest first


def core_gendered_form(match_obj, pattern_name):
    """Return the clean (stem, core_form) pair with any compound
    continuation past the innen/in suffix stripped off."""
    text = match_obj.group(0)
    if pattern_name == 'paired':
        return match_obj.group(1), text
    if pattern_name in ('asterisk', 'colon', 'underscore'):
        sep = {'asterisk': '*', 'colon': ':', 'underscore': '_'}[pattern_name]
        stem, _, rest = text.partition(sep)
        rest_lower = rest.lower()
        for suf in _SUFFIX_CANDIDATES:
            if rest_lower.startswith(suf):
                return stem, stem + sep + rest[:len(suf)]
        return stem, text
    if pattern_name == 'binnen_i':
        idx = text.index('I', 1)
        stem = text[:idx]
        rest = text[idx + 1:]
        rest_lower = rest.lower()
        for suf in ['nnen', 'n']:  # completes "Innen" or bare "In"
            if rest_lower.startswith(suf):
                return stem, text[:idx + 1] + rest[:len(suf)]
        return stem, text
    raise ValueError(f"unknown pattern: {pattern_name}")


def is_html(path):
    try:
        with open(path, 'rb') as f:
            head = f.read(512).lower()
        return b'<!doctype' in head or b'<html' in head
    except OSError:
        return False


def pairs_from_file(path, all_files):
    if all_files and not path.lower().endswith('.html') and not is_html(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
    except Exception:
        return []

    out = []
    for name, pattern in GENDER_PATTERNS.items():
        for m in pattern.finditer(text):
            stem, core = core_gendered_form(m, name)
            out.append((stem, name, m.group(0), core))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('corpus_dirs', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--progress-every', type=int, default=5000)
    args = parser.parse_args()

    counts = Counter()  # (stem, convention, raw_form, core_form) -> count
    for corpus_dir in args.corpus_dirs:
        source_name = os.path.basename(os.path.normpath(corpus_dir))
        print(f"Scanning {corpus_dir} ...")
        n_files = 0
        for dirpath, _, filenames in os.walk(corpus_dir):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                for quad in pairs_from_file(path, all_files=True):
                    counts[quad] += 1
                n_files += 1
                if n_files % args.progress_every == 0:
                    print(f"  [{source_name}] {n_files:,} files scanned, "
                          f"{len(counts):,} unique (stem,convention,form) triples so far ...")
        print(f"  [{source_name}] done: {n_files:,} files scanned.")

    print(f"\nTotal unique (stem, convention, raw_form, core_form) records: {len(counts):,}")
    n_contaminated = sum(n for (s, c, raw, core), n in counts.items() if raw != core)
    n_total = sum(counts.values())
    print(f"Compound-continuation contamination: {n_contaminated:,}/{n_total:,} "
          f"raw matches ({n_contaminated / n_total:.1%}) had a compound tail past the innen/in suffix.")

    records = [
        {'stem': stem, 'convention': conv, 'raw_form': raw, 'core_form': core, 'count': n}
        for (stem, conv, raw, core), n in counts.items()
    ]
    records.sort(key=lambda r: -r['count'])

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
