"""
Extract BIO-labeled training data for German gender term annotation from an HTML corpus.

For each sentence, tokens are labeled O, B-GENDER, or I-GENDER.
Negative (non-gendered) sentences are included at a configurable ratio.

Key design decisions (informed by iterative regex development in ai_chats/):

1. Regex patterns use explicit German character classes [a-zA-ZäöüÄÖÜß] instead of \\w.
   \w matches digits and underscores which produce false positives in German newspaper
   text.  The asterisk pattern additionally uses [^\*\s] to ensure exactly one asterisk.
   The paired form uses a backreference (\1) so only the exact base+innen pair matches.

2. Train/valid/test splits are performed at ARTICLE level, not sentence level.
   Splitting sentences randomly would let sentences from the same article appear in
   both training and test sets (data leakage).  Article-level splitting prevents this.

Output: JSONL files, one example per line:
    {"tokens": ["Die", "Lehrer:innen", "kommen"], "labels": ["O", "B-GENDER", "O"]}

Usage:
    python 01_extract_training_data.py ../corpora/taz.de data/
    python 01_extract_training_data.py ../corpora/taz.de data/ --max-positive 100000 --seed 42
"""

import os
import re
import json
import random
import argparse
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Regex patterns for German gender forms.
#
# All word-character classes use [a-zA-ZäöüÄÖÜß] rather than \w to avoid
# matching digits and underscores that are not part of German words.
#
# Developed through iterative refinement (see ai_chats/deepseek-chat.2025-07-13.pdf):
#   - Paired:     backreference \1 ensures the stem before "und" and after match exactly
#   - Asterisk:   [^\*\s]+ on both sides ensures exactly one * and no whitespace inside
#   - Colon/underscore: capital-first stem requirement filters out URLs, timestamps, etc.
#   - Binnen-I:   capital I mid-word after a lower-case run
# ---------------------------------------------------------------------------
_ALPHA = r'[a-zA-ZäöüÄÖÜß]'
_ALPHA_LOWER = r'[a-zäöüß]'
_ALPHA_UPPER = r'[A-ZÄÖÜ]'

GENDER_PATTERNS = {
    # "Lehrer und Lehrerinnen" / "Lehrer und LehrerInnen"
    'paired':    re.compile(
        r'\b(' + _ALPHA + r'{3,})\s+und\s+\1(?:innen|Innen)\b'
    ),
    # "Lehrer*innen" — exactly one asterisk, German chars only on both sides
    'asterisk':  re.compile(
        r'\b' + _ALPHA_UPPER + r'[^\*\s]{2,}\*' + _ALPHA_LOWER + r'{2,}\b'
    ),
    # "Lehrer:innen"
    'colon':     re.compile(
        r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}:' + _ALPHA_LOWER + r'{2,}\b'
    ),
    # "Lehrer_innen"
    'underscore':re.compile(
        r'\b' + _ALPHA_UPPER + _ALPHA + r'{3,}_' + _ALPHA_LOWER + r'{2,}\b'
    ),
    # "LeserInnen" — capital I after a lower-case run
    'binnen_i':  re.compile(
        r'\b' + _ALPHA_UPPER + _ALPHA_LOWER + r'{2,}I' + _ALPHA_LOWER + r'{2,}\b'
    ),
}

# Sentence splitter: the lookbehind approach from the ai_chats avoids breaking on
# common abbreviations like "Dr.", "bzw.", single-letter initials, etc.
_SENT_SPLIT = re.compile(
    r'(?<!\w\.\w.)(?<![A-ZÄÖÜ][a-zäöüß]\.)(?<=\.|\?|!)\s+'
)


def has_gender_term(sentence):
    return any(p.search(sentence) for p in GENDER_PATTERNS.values())


def tokenize_and_label(sentence):
    """Split sentence on whitespace and assign BIO labels for gender spans."""
    words = sentence.split()
    if not words:
        return [], []

    # Compute character offsets for each whitespace token.
    offsets = []
    pos = 0
    for word in words:
        start = sentence.find(word, pos)
        offsets.append((start, start + len(word)))
        pos = start + len(word)

    # Collect all gender spans (may overlap for different patterns; union them).
    gender_spans = []
    for pattern in GENDER_PATTERNS.values():
        for m in pattern.finditer(sentence):
            gender_spans.append((m.start(), m.end()))

    # BIO labelling: first overlapping token → B-GENDER, rest → I-GENDER.
    labels = ['O'] * len(words)
    for span_start, span_end in gender_spans:
        first_in_span = True
        for i, (word_start, word_end) in enumerate(offsets):
            if word_start < span_end and word_end > span_start:
                labels[i] = 'B-GENDER' if first_in_span else 'I-GENDER'
                first_in_span = False

    return words, labels


def is_html(path):
    """Return True if the file starts with an HTML doctype or root tag.

    Used to handle corpora like www.woz.ch where articles are stored without
    a .html extension.
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(512).lower()
        return b'<!doctype' in head or b'<html' in head
    except OSError:
        return False


def extract_article_sentences(html_path):
    """Return (article_text, [sentences]) from a single HTML file.

    article_text is the cleaned plain text of the whole article.
    sentences is a list of individual sentence strings (20–500 chars).
    Returns (None, []) on error.
    """
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
    """Walk corpus_dir and collect per-article sentence examples.

    Returns a list of article dicts:
        {
            'path':     str,
            'positive': [{'tokens': [...], 'labels': [...]}],
            'negative': [{'tokens': [...], 'labels': ['O', ...]}],
        }

    Only articles that have at least one sentence (positive or negative) are
    included.  Collecting at article level enables leak-free splitting later.

    all_files: if True, attempt to parse every file as HTML (needed for corpora
    like www.woz.ch where articles are stored without a .html extension).
    """
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
                # For extensionless files sniff the content before parsing.
                if not is_html(path):
                    continue
            _, sentences = extract_article_sentences(path)

            pos_examples, neg_examples = [], []
            for sentence in sentences:
                if has_gender_term(sentence):
                    if total_pos < max_positive:
                        tokens, labels = tokenize_and_label(sentence)
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
    """Flatten and shuffle examples from a list of article dicts."""
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
        description='Extract BIO-labeled training data for gender term annotation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('corpus_dir', help='Root directory containing HTML files')
    parser.add_argument('output_dir', help='Directory to write train/valid/test JSONL files')
    parser.add_argument('--max-positive', type=int, default=50000,
                        help='Maximum number of positive (gendered) examples to collect')
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

    # Split at article level to prevent data leakage between splits.
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
