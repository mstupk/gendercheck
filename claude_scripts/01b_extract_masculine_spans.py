"""
Extract BIO-labeled training data for Stage 1 of the generation pipeline
(SPECIFICATION.md §3.2): MASCULINE-plural candidate span detection.

This is deliberately NOT the same task as 01_extract_training_data.py, even
though it shares the exact same corpus, regexes, and JSONL schema
({"tokens": [...], "labels": [...]}  with O/B-GENDER/I-GENDER). That script
finds spans that are ALREADY gendered (e.g. "Lehrer:innen") -- useful as a
"don't touch already-gendered text" guard, but not the thing the autocorrect
feature actually needs, which is: given ordinary masculine-generic text
("Lehrer"), find the word that SHOULD be offered for gendering.

This script produces that instead, by degendering every mined gendered span
back to its masculine base form (same substitution rules as
scripts/retranslator_4.py's translate_matched_term(), reimplemented against
the umlaut-safe GENDER_PATTERNS below rather than retranslator's own looser
\\w-based regexes -- see 01_extract_training_data.py's docstring for why that
distinction matters) and labeling the SUBSTITUTED masculine word as the
positive span instead of the original gendered word.

Example:
    original sentence: "Die Lehrer:innen kommen morgen."
    positive example :  tokens=["Die","Lehrer","kommen","morgen."]
                         labels=["O","B-GENDER","O","O"]

Negative examples (sentences with no gendered term found) are copied through
completely unmodified, exactly as in 01_extract_training_data.py -- they are
real sentences and are exactly the "nothing to gender here" signal Stage 2
(SPECIFICATION.md §3.2) will also need.

Output goes to a directory clearly separate from the Approach A/B data
(data/, data_merged/, data_taz.de/, data_www.woz.ch/) so the two datasets are
never confused -- see run_stage1_pipeline.sh, which writes to
claude_pipeline_output_stage1/ by convention.

Usage (identical CLI to 01_extract_training_data.py, so it's a drop-in swap):
    python 01b_extract_masculine_spans.py ../corpora/taz.de data_stage1_taz.de/
    python 01b_extract_masculine_spans.py ../corpora/taz.de data/ --max-positive 10000 --seed 42
"""

import os
import re
import json
import random
import argparse
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Identical to 01_extract_training_data.py -- see that file for the rationale
# (umlaut-safe classes, backreference for paired forms, capital-stem
# requirement to filter URLs/timestamps, etc.). Kept as a literal copy rather
# than a shared import so this script has no cross-file dependency and stays
# a single, self-contained artifact like its sibling extraction scripts.
# ---------------------------------------------------------------------------
_ALPHA = r'[a-zA-ZäöüÄÖÜß]'
_ALPHA_LOWER = r'[a-zäöüß]'
_ALPHA_UPPER = r'[A-ZÄÖÜ]'

GENDER_PATTERNS = {
    'paired':    re.compile(
        r'\b(' + _ALPHA + r'{3,})\s+und\s+\1(?:innen|Innen)\b'
    ),
    'asterisk':  re.compile(
        r'\b' + _ALPHA_UPPER + r'[^\*\s]{2,}\*' + _ALPHA_LOWER + r'{2,}\b'
    ),
    'colon':     re.compile(
        r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}:' + _ALPHA_LOWER + r'{2,}\b'
    ),
    'underscore':re.compile(
        r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}_' + _ALPHA_LOWER + r'{2,}\b'
    ),
    'binnen_i':  re.compile(
        r'\b' + _ALPHA_UPPER + _ALPHA_LOWER + r'{2,}I' + _ALPHA_LOWER + r'{2,}\b'
    ),
}

_SENT_SPLIT = re.compile(
    r'(?<!\w\.\w.)(?<![A-ZÄÖÜ][a-zäöüß]\.)(?<=\.|\?|!)\s+'
)


def has_gender_term(sentence):
    return any(p.search(sentence) for p in GENDER_PATTERNS.values())


def find_gender_matches(sentence):
    """Return non-overlapping (start, end, pattern_name, match_obj) tuples
    across all five patterns, sorted by position. On the rare overlap
    between two patterns' matches, the earlier-starting one wins."""
    candidates = []
    for name, pattern in GENDER_PATTERNS.items():
        for m in pattern.finditer(sentence):
            candidates.append((m.start(), m.end(), name, m))
    candidates.sort(key=lambda c: c[0])

    accepted = []
    last_end = -1
    for start, end, name, m in candidates:
        if start >= last_end:
            accepted.append((start, end, name, m))
            last_end = end
    return accepted


def degender_match(match_obj, pattern_name):
    """Return the masculine-plural base form for one matched gendered span.

    Same substitution rule per pattern as scripts/retranslator_4.py's
    translate_matched_term():
      paired      "Lehrer und Lehrerinnen" -> group(1) -> "Lehrer"
      asterisk    "Lehrer*innen"           -> text before "*" -> "Lehrer"
      colon       "Lehrer:innen"           -> text before ":" -> "Lehrer"
      underscore  "Lehrer_innen"           -> text before "_" -> "Lehrer"
      binnen_i    "LeserInnen"             -> text before the embedded
                                              capital I -> "Leser"
                  (the pattern guarantees >=2 lowercase chars after the
                  word-initial capital before the embedded I, so searching
                  for "I" from index 1 can't false-hit the initial letter)
    """
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


def degender_and_label(sentence):
    """Replace every gendered span in `sentence` with its masculine base
    form and BIO-label the substituted masculine word(s).

    Returns (tokens, labels), or None if no gendered span was found (caller
    should treat that the same as has_gender_term() == False).
    """
    matches = find_gender_matches(sentence)
    if not matches:
        return None

    result_parts = []
    ins_spans = []  # (start_char, end_char) of each masculine substitution, in the *new* string
    last_end = 0
    for start, end, name, m in matches:
        result_parts.append(sentence[last_end:start])
        insertion_start = sum(len(p) for p in result_parts)
        masculine = degender_match(m, name)
        result_parts.append(masculine)
        ins_spans.append((insertion_start, insertion_start + len(masculine)))
        last_end = end
    result_parts.append(sentence[last_end:])
    degendered = ''.join(result_parts)

    words = degendered.split()
    if not words:
        return [], []

    offsets = []
    pos = 0
    for word in words:
        wstart = degendered.find(word, pos)
        offsets.append((wstart, wstart + len(word)))
        pos = wstart + len(word)

    labels = ['O'] * len(words)
    for span_start, span_end in ins_spans:
        first_in_span = True
        for i, (wstart, wend) in enumerate(offsets):
            if wstart < span_end and wend > span_start:
                labels[i] = 'B-GENDER' if first_in_span else 'I-GENDER'
                first_in_span = False

    return words, labels


def is_html(path):
    try:
        with open(path, 'rb') as f:
            head = f.read(512).lower()
        return b'<!doctype' in head or b'<html' in head
    except OSError:
        return False


def extract_article_sentences(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
            soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        sentences = _SENT_SPLIT.split(text)
        sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 500]
        return text, sentences
    except Exception as e:
        print(f"Warning: could not process {html_path}: {e}")
        return None, []


def collect_articles(corpus_dir, max_positive, max_neg, all_files=False):
    articles = []
    total_pos = 0
    total_neg = 0

    for dirpath, _, filenames in os.walk(corpus_dir):
        if total_pos >= max_positive and total_neg >= max_neg:
            break
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if not all_files and not filename.lower().endswith('.html'):
                continue
            if all_files and filename.lower().endswith('.html') is False:
                if not is_html(path):
                    continue
            _, sentences = extract_article_sentences(path)

            pos_examples, neg_examples = [], []
            for sentence in sentences:
                if has_gender_term(sentence):
                    if total_pos < max_positive:
                        result = degender_and_label(sentence)
                        if result:
                            tokens, labels = result
                            if tokens:
                                pos_examples.append({'tokens': tokens, 'labels': labels})
                                total_pos += 1
                else:
                    if total_neg < max_neg:
                        words = sentence.split()
                        if words:
                            neg_examples.append({
                                'tokens': words,
                                'labels': ['O'] * len(words),
                            })
                            total_neg += 1

            if pos_examples or neg_examples:
                articles.append({
                    'path':     path,
                    'positive': pos_examples,
                    'negative': neg_examples,
                })

        if total_pos > 0 and total_pos % 5000 < 50:
            print(f"  {total_pos:,} positive, {total_neg:,} negative from "
                  f"{len(articles):,} articles ...")

    return articles, total_pos, total_neg


def articles_to_examples(article_list):
    examples = []
    for art in article_list:
        examples.extend(art['positive'])
        examples.extend(art['negative'])
    random.shuffle(examples)
    return examples


def write_split(examples, path):
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(examples):,} examples → {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract BIO-labeled training data for Stage 1 masculine-span detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('corpus_dir', help='Root directory containing HTML files')
    parser.add_argument('output_dir', help='Directory to write train/valid/test JSONL files')
    parser.add_argument('--max-positive', type=int, default=50000,
                        help='Maximum number of positive (masculine-candidate) examples to collect')
    parser.add_argument('--neg-ratio', type=float, default=1.0,
                        help='Ratio of negative examples to positive examples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--valid-ratio', type=float, default=0.1)
    parser.add_argument('--all-files', action='store_true',
                        help='Try to parse all files as HTML, not just *.html '
                             '(required for corpora like www.woz.ch where articles '
                             'have no file extension)')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    max_neg = int(args.max_positive * args.neg_ratio)

    print(f"Scanning {args.corpus_dir} ...")
    articles, total_pos, total_neg = collect_articles(
        args.corpus_dir, args.max_positive, max_neg, all_files=args.all_files
    )
    print(f"Total: {total_pos:,} positive, {total_neg:,} negative "
          f"examples across {len(articles):,} articles.")

    random.shuffle(articles)
    n = len(articles)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)

    train_articles = articles[:n_train]
    valid_articles = articles[n_train:n_train + n_valid]
    test_articles  = articles[n_train + n_valid:]

    print(f"Article split: {len(train_articles)} train / "
          f"{len(valid_articles)} valid / {len(test_articles)} test")

    print("Writing splits ...")
    write_split(articles_to_examples(train_articles), os.path.join(args.output_dir, 'train.jsonl'))
    write_split(articles_to_examples(valid_articles), os.path.join(args.output_dir, 'valid.jsonl'))
    write_split(articles_to_examples(test_articles),  os.path.join(args.output_dir, 'test.jsonl'))
    print("Done.")


if __name__ == '__main__':
    main()
